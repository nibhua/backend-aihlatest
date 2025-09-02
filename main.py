from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from indexer.config import BASE_STORE_DIR
import uuid
import os
import shutil
import time
import json
import sys
import subprocess
from pathlib import Path
import re
import difflib
import unicodedata  # <— for canonical filename matching

# Import logging components
from core import (
    APILoggingMiddleware, TimeoutMiddleware, api_logger_instance, get_request_id,
    setup_exception_handlers
)
from core.cleanup import setup_cleanup_handlers, setup_signal_handlers
from core.workspace_manager import workspace_manager
from core.concurrency_decorators import (
    build_concurrency, query_concurrency, snippet_concurrency
)
from core.concurrency_manager import get_concurrency_status

# Import service routers
from podcast_service.router import router as podcast_router
from insights_service.router import router as insights_router
from chat_service.router import router as chat_router
from relevance_service.router import router as relevance_router
from collection_summary_service.router import router as collection_summary_router

app = FastAPI(
    title="Document Insight & Engagement System",
    version="1.0.0",
    description="Enhanced PDF retrieval system with insights, chat, podcast, and relevance analysis"
)

# Add API logging middleware first
app.add_middleware(APILoggingMiddleware)

# Add timeout middleware (150 seconds)
app.add_middleware(TimeoutMiddleware, timeout=150.0)

# Setup global exception handlers
setup_exception_handlers(app)

# Setup cleanup handlers
setup_cleanup_handlers(app)
setup_signal_handlers()

# CORS: allow every origin (no credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include service routers
app.include_router(podcast_router)
app.include_router(insights_router)
app.include_router(chat_router)
app.include_router(relevance_router)
app.include_router(collection_summary_router)

# Optional quick-test UI at /ui
app.mount("/ui", StaticFiles(directory="web_static", html=True), name="static")

# Paths
INPUT_DIR = Path("uploads")          # where files are uploaded from frontend
VECTOR_DIR = BASE_STORE_DIR          # your existing vector store root
UPLOAD_DIR = Path("uploads")         # optional

# Ensure folders exist
for p in (INPUT_DIR, VECTOR_DIR, UPLOAD_DIR):
    p.mkdir(parents=True, exist_ok=True)


def _clear_dir(p: Path):
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        return
    for child in p.iterdir():
        if child.is_file() or child.is_symlink():
            try:
                child.unlink(missing_ok=True)
            except Exception:
                pass
        else:
            shutil.rmtree(child, ignore_errors=True)


def _clear_collection_folders():
    """Clear all col_ folders in vector_store while preserving current_collection.txt"""
    if not VECTOR_DIR.exists():
        return
    
    for child in VECTOR_DIR.iterdir():
        if child.is_dir() and child.name.startswith("col_"):
            try:
                shutil.rmtree(child, ignore_errors=True)
            except Exception:
                pass


def _latest_collection_id() -> str | None:
    subs = [d for d in VECTOR_DIR.iterdir() if d.is_dir() and d.name.startswith("col_")]
    if not subs:
        return None
    latest = max(subs, key=lambda d: d.stat().st_mtime)
    return latest.name


@app.post("/collections/build")
@build_concurrency
async def build_collection(request: Request, files: List[UploadFile] = File(...)):
    """
    Accepts PDFs, creates an isolated workspace, runs the build pipeline,
    and returns collection details.
    """
    start = time.time()
    job_id = str(uuid.uuid4())
    request_id = getattr(request.state, 'request_id', get_request_id())

    # Log build start (basic info only)
    api_logger_instance.log_performance(
        request_id=request_id,
        operation="build_collection_start",
        duration=0,
        files_count=len(files)
    )

    # Generate collection ID from file names
    from indexer.nn_store import make_collection_id
    file_names = [f.filename for f in files]
    collection_id = make_collection_id(file_names)
    
    # Create isolated workspace for this collection
    workspace_path = workspace_manager.create_workspace(collection_id)
    uploads_dir = workspace_manager.get_uploads_path(collection_id)
    
    # Save uploaded files to the workspace uploads directory
    saved_files = []
    for uf in files:
        dest = uploads_dir / uf.filename
        with dest.open("wb") as f:
            f.write(await uf.read())
        saved_files.append(uf.filename)

    # Run the build script with the workspace uploads directory
    cmd = [sys.executable, "-m", "scripts.build_collection", "--input_dir", str(uploads_dir), "--collection_id", collection_id]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        # Clean up workspace on failure
        workspace_manager.cleanup_workspace(collection_id)
        return JSONResponse(
            status_code=500,
            content={
                "job_id": job_id,
                "status": "failed",
                "message": "Build process error",
                "stderr_tail": (e.stderr or "")[-2000:],
                "stdout_tail": (e.stdout or "")[-500:],
            },
        )

    # 5) Read counts from meta.json / chunks.json (best-effort)
    docs = None
    chunks = None
    vector_store_dir = workspace_manager.get_vector_store_path(collection_id)
    meta_path = vector_store_dir / "meta.json"
    chunks_path = vector_store_dir / "chunks.json"

    try:
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
                docs = meta.get("num_docs") or meta.get("docs") or meta.get("document_count")
        if chunks_path.exists():
            with chunks_path.open("r", encoding="utf-8") as f:
                cj = json.load(f)
                if isinstance(cj, list):
                    chunks = len(cj)
                    if not docs:
                        try:
                            docs = len({c.get("doc_id") for c in cj if "doc_id" in c})
                        except Exception:
                            pass
                elif isinstance(cj, dict) and "chunks" in cj:
                    clist = cj["chunks"]
                    chunks = len(clist)
                    if not docs:
                        try:
                            docs = len({c.get("doc_id") for c in clist if "doc_id" in c})
                        except Exception:
                            pass
    except Exception:
        pass

    docs = int(docs or 0)
    chunks = int(chunks or 0)

    # 6) Collection isolation is now handled via workspace_manager
    # No need to write to CURRENT_PTR since each collection is isolated

    duration = round(time.time() - start, 2)

    # Log build completion
    api_logger_instance.log_performance(
        request_id=request_id,
        operation="build_collection_complete",
        duration=duration,
        collection_id=collection_id
    )

    response_data = {
        "job_id": job_id,
        "status": "done",
        "collection_id": collection_id,
        "docs": docs,
        "chunks": chunks,
        "duration_sec": duration,
        "saved_files": saved_files,
        "message": "Build complete",
    }

    return JSONResponse(content=response_data)


# -----------------------------
# helpers for bbox lookup  (UPDATED)
# -----------------------------

def _canon_filename(s: str) -> str:
    """lowercase, strip, basename, drop diacritics, collapse non-alnum to single space"""
    s = os.path.basename(str(s or ""))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def _parse_bbox(c: Dict[str, Any]) -> list | None:
    """
    Accept common shapes:
      - c['bbox'] = [x0,y0,x1,y1]  (assumed BL after builder fix)
      - c['boundingBox'] = [x0,y0,x1,y1]  OR  {'left':..,'top':..,'width':..,'height':..}
    """
    for k in ("bbox", "bbox_pdf", "bbox_xyxy", "boundingBox"):
        v = c.get(k)
        if isinstance(v, list) and len(v) == 4:
            return [float(v[0]), float(v[1]), float(v[2]), float(v[3])]
        if isinstance(v, dict):
            if {"left", "top", "width", "height"} <= set(v.keys()):
                left = float(v["left"]); top = float(v["top"])
                width = float(v["width"]); height = float(v["height"])
                # We assume incoming coords are already bottom-left based; return xyxy.
                return [left, top, left + width, top + height]
    return None


def _load_chunks(collection_id: str) -> List[Dict[str, Any]]:
    """
    Return a flat list of {
        doc_id, doc_id_canon, page, heading_text, bbox,
        page_height_pt, bbox_origin,
        content_regions?, content_quads?
    }
    Pages are normalized to 1-based if needed.
    """
    # Determine chunks path - try workspace-based first, then legacy
    chunks_path = None
    
    if collection_id.startswith('col_'):
        try:
            vector_store_dir = workspace_manager.get_vector_store_path(collection_id)
            potential_path = vector_store_dir / "chunks.json"
            if potential_path.exists():
                chunks_path = potential_path
        except Exception:
            pass  # Fall back to legacy path
    
    # Fall back to legacy path if workspace path not found
    if chunks_path is None:
        potential_path = VECTOR_DIR / collection_id / "chunks.json"
        if potential_path.exists():
            chunks_path = potential_path
    
    if chunks_path is None or not chunks_path.exists():
        return []
    
    try:
        with chunks_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "chunks" in data:
            data = data["chunks"]
        if not isinstance(data, list):
            return []

        # detect 0-based if any page==0
        pages = []
        for c in data:
            if isinstance(c, dict) and "page" in c:
                try:
                    pages.append(int(c.get("page")))
                except Exception:
                    pass
        zero_based = any(p == 0 for p in pages)

        out: List[Dict[str, Any]] = []
        for c in data:
            if not isinstance(c, dict):
                continue
            raw_doc = c.get("doc_id") or c.get("source") or c.get("file_name") or ""
            page_raw = c.get("page") or c.get("page_number") or c.get("pageIndex")
            try:
                page = int(page_raw)
            except Exception:
                continue
            if zero_based and page >= 0:
                page = page + 1  # normalize to 1-based

            bbox = _parse_bbox(c)
            heading = c.get("heading_text") or c.get("heading") or c.get("title") or ""
            page_h = c.get("page_height_pt")
            try:
                page_h = float(page_h) if page_h is not None else None
            except Exception:
                page_h = None

            # NEW: pass through content_regions / content_quads if present
            content_regions = c.get("content_regions") if isinstance(c.get("content_regions"), (dict,)) else None
            content_quads = c.get("content_quads") if isinstance(c.get("content_quads"), (dict,)) else None

            doc_id = os.path.basename(str(raw_doc))
            out.append({
                "doc_id": doc_id,
                "doc_id_canon": _canon_filename(doc_id),
                "page": page,
                "heading_text": heading,
                "bbox": bbox,
                "page_height_pt": page_h,
                "bbox_origin": c.get("bbox_origin") or None,
                "content_regions": content_regions,
                "content_quads": content_quads,
            })
        return out
    except Exception:
        return []


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _best_chunk_for(chunks: List[Dict[str, Any]], file_name: str, page: int, heading: str) -> Optional[Dict[str, Any]]:
    """
    Find best matching chunk for (file_name, page, heading):
      - compares canonicalized basenames
      - tries exact page, then +/-1..2
      - picks best heading similarity when multiple
    """
    target_name = _canon_filename(file_name)
    page_candidates = [page, page - 1, page + 1, page - 2, page + 2]

    def sim(a, b): return difflib.SequenceMatcher(a=a, b=b).ratio()

    cand = [c for c in chunks if c["doc_id_canon"] == target_name and c["page"] in page_candidates]
    if not cand:
        # last-chance fuzzy on filename (typos, underscores vs spaces, etc.)
        cand = [c for c in chunks if sim(c["doc_id_canon"], target_name) >= 0.9 and c["page"] in page_candidates]
    if not cand:
        return None

    with_bbox = [c for c in cand if isinstance(c.get("bbox"), list)]
    if not with_bbox:
        return None
    if len(with_bbox) == 1:
        return with_bbox[0]

    # score by heading similarity
    h = _norm(heading)
    best, best_score = None, -1.0
    for c in with_bbox:
        ch = _norm(c.get("heading_text") or "")
        score = difflib.SequenceMatcher(a=h, b=ch).ratio()
        if score > best_score:
            best, best_score = c, score
    return best


def _areas_for_page(hit_chunk: Dict[str, Any], page: int) -> List[List[float]]:
    """
    Extract BL-origin rectangles for the given page from content_regions (if present).
    JSON stores keys as strings; accept str/int keys gracefully.
    """
    cr = hit_chunk.get("content_regions")
    if not isinstance(cr, dict):
        return []
    # try string key first, then int key
    key_str = str(page)
    if key_str in cr and isinstance(cr[key_str], list):
        # ensure numeric floats
        out = []
        for rect in cr[key_str]:
            if isinstance(rect, (list, tuple)) and len(rect) == 4:
                try:
                    out.append([float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])])
                except Exception:
                    pass
        return out
    if page in cr and isinstance(cr[page], list):
        out = []
        for rect in cr[page]:
            if isinstance(rect, (list, tuple)) and len(rect) == 4:
                try:
                    out.append([float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])])
                except Exception:
                    pass
        return out
    return []


def _quads_for_page(hit_chunk: Dict[str, Any], page: int) -> List[List[float]]:
    """
    Optional future-proofing: if you add line-precise content_quads in the builder,
    this will surface them. Each quad is 8 numbers (xul,yul, xur,yur, xll,yll, xlr,ylr).
    """
    cq = hit_chunk.get("content_quads")
    out: List[List[float]] = []
    if not isinstance(cq, dict):
        return out
    key_str = str(page)
    arr = cq.get(key_str) or cq.get(page)
    if isinstance(arr, list):
        for q in arr:
            if isinstance(q, (list, tuple)) and len(q) == 8:
                try:
                    out.append([float(v) for v in q])
                except Exception:
                    pass
    return out


# -----------------------------
# REAL QUERY ENDPOINT
# -----------------------------

# Example CLI line:
# " 1. score= 0.016  p.2    1 teaspoon paprika  (Dinner Ideas - Mains_1.pdf)"
_LINE_RE = re.compile(
    r"^\s*(\d+)\.\s+score=\s*([0-9.]+)\s+p\.(\d+)\s+(.+?)\s+\((.+)\)\s*$"
)

@app.post("/query")
@query_concurrency
async def query_collection(
    request: Request,
    collection_id: str = Form(...),      # REQUIRED: collection ID to query
    persona: str = Form(""),
    job: str = Form(""),
    mode: str = Form("legacy"),          # "legacy" | "context"
    k: int = Form(10),                   # ignored; we always return Top-10
    query_text: str = Form(""),          # used when mode="context"
):
    """
    Unified query endpoint:
      - mode="legacy": persona+job
      - mode="context": free-text (e.g., user's selected text)
    ALWAYS returns up to Top-10 results, regardless of requested k.
    """
    t0 = time.time()
    request_id = getattr(request.state, 'request_id', get_request_id())

    # Log query start
    api_logger_instance.log_performance(
        request_id=request_id,
        operation="query_start",
        duration=0,
        mode=mode,
        persona=persona,
        job=job,
        query_text_length=len(query_text) if query_text else 0
    )

    # Validate collection_id parameter
    if not collection_id or not collection_id.strip():
        raise HTTPException(status_code=400, detail="collection_id parameter is required.")
    
    collection_id = collection_id.strip()
    
    # Verify the collection exists
    if not workspace_manager.workspace_exists(collection_id):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_id}' not found. Build the collection first.")

    # Enforce Top-10 regardless of what client sends
    k_effective = 10

    cmd = [sys.executable, "-m", "scripts.query_collection", "--collection", collection_id]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    mode = (mode or "legacy").strip().lower()
    if mode == "context":
        if not query_text.strip():
            raise HTTPException(status_code=400, detail="query_text is required when mode='context'")
        cmd += ["text", "--query", query_text, "--k", str(k_effective), "--mmr"]
        if persona.strip():
            cmd += ["--persona", persona]
        if job.strip():
            cmd += ["--job", job]
    else:
        if not (persona.strip() or job.strip()):
            raise HTTPException(status_code=400, detail="persona or job is required when mode='legacy'")
        cmd += ["persona", "--persona", persona, "--job", job, "--k", str(k_effective), "--mmr"]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        stdout = proc.stdout or ""
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Query process error",
                "stderr_tail": (e.stderr or "")[-2000:],
                "stdout_tail": (e.stdout or "")[-500:],
            },
        )

    # Parse up to k_effective hits from CLI output
    results: List[Dict[str, Any]] = []
    lines = stdout.splitlines()

    i = 0
    while i < len(lines) and len(results) < k_effective:
        m = _LINE_RE.match(lines[i])
        if m:
            try:
                rank = int(m.group(1))
                score = float(m.group(2))
                page = int(m.group(3))
                heading = m.group(4).strip()
                file_name = m.group(5).strip()
            except Exception:
                i += 1
                continue

            snippet = ""
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                snippet = nxt.lstrip("↳").strip()

            results.append({
                "rank": rank,
                "doc_id": file_name,
                "file_name": file_name,
                "page": page,
                "heading": heading,
                "snippet": snippet,
                "score": round(score, 3),
                "collection_id": collection_id,  # Add collection_id to each result
            })

            i += 2
            continue

        i += 1

    # Attach bbox + page_height_pt + areas/quads from chunks.json (best-effort)
    chunks = _load_chunks(collection_id)
    if chunks:
        for r in results:
            hit = _best_chunk_for(chunks, r["file_name"], r["page"], r["heading"])
            if hit and isinstance(hit.get("bbox"), list):
                r["bbox"] = hit["bbox"]              # BL coords
                if hit.get("page_height_pt") is not None:
                    r["page_height_pt"] = hit["page_height_pt"]
                if hit.get("bbox_origin"):
                    r["bbox_origin"] = hit["bbox_origin"]

                areas = _areas_for_page(hit, r["page"])
                if areas:
                    r["areas"] = areas  # [[x0,y0,x1,y1], ...] in BL origin
                quads = _quads_for_page(hit, r["page"])
                if quads:
                    r["quads"] = quads  # [8-number arrays] in BL origin
            else:

                print(
                    f"[bbox-miss] file={r['file_name']} page={r['page']} heading={r['heading']!r}",
                    flush=True
                )

    latency_ms = int((time.time() - t0) * 1000)

    # Log query completion
    api_logger_instance.log_performance(
        request_id=request_id,
        operation="query_complete",
        duration=latency_ms / 1000,
        results_count=len(results),
        collection_id=collection_id
    )

    response_data = {
        "collection_id": collection_id,
        "persona": persona,
        "job": job,
        "mode": mode,
        "k": k_effective,           # always 10
        "query_text": query_text,
        "results": results,
        "latency_ms": latency_ms
    }

    return JSONResponse(content=response_data)


# -------------------------------------------
# NEW: Enhanced snippet endpoint for text selection
# -------------------------------------------
@app.post("/snippets")
@snippet_concurrency
async def get_snippets(request: Request, collection_id: str = Form(...), selected_text: str = Form(...)):
    """
    Get relevant snippets based on selected text from documents.
    This is the core functionality for the text selection user journey.
    """
    t0 = time.time()
    request_id = getattr(request.state, 'request_id', get_request_id())

    # Log snippet search start
    api_logger_instance.log_performance(
        request_id=request_id,
        operation="snippet_search_start",
        duration=0,
        selected_text_length=len(selected_text)
    )

    # Validate collection_id parameter
    if not collection_id or not collection_id.strip():
        raise HTTPException(status_code=400, detail="collection_id parameter is required.")
    
    collection_id = collection_id.strip()
    
    # Verify the collection exists
    if not workspace_manager.workspace_exists(collection_id):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_id}' not found. Build the collection first.")

    # Use the existing query mechanism but with selected text as the query
    cmd = [
        sys.executable, "-m", "scripts.query_collection",
        "--collection", collection_id,
        "persona",
        "--persona", "researcher",  # Default persona for snippet search
        "--job", selected_text,     # Use selected text as the job/query
        "--mmr",
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        stdout = proc.stdout or ""
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Snippet search error",
                "stderr_tail": (e.stderr or "")[-2000:],
                "stdout_tail": (e.stdout or "")[-500:],
            },
        )

    results: List[Dict[str, Any]] = []
    lines = stdout.splitlines()

    i = 0
    while i < len(lines) and len(results) < 10:  # Get more results for snippets
        m = _LINE_RE.match(lines[i])
        if m:
            try:
                rank = int(m.group(1))
                score = float(m.group(2))
                page = int(m.group(3))
                heading = m.group(4).strip()
                file_name = m.group(5).strip()
            except Exception:
                i += 1
                continue

            snippet = ""
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                snippet = nxt.lstrip("↳").strip()

            results.append({
                "rank": rank,
                "doc_id": file_name,
                "file_name": file_name,
                "page": page,
                "heading": heading,
                "snippet": snippet,
                "score": round(score, 3),
                "collection_id": collection_id,  # Add collection_id to snippets results
            })

            i += 2
            continue

        i += 1

    # Attach bbox information
    chunks = _load_chunks(collection_id)
    if chunks:
        for r in results:
            hit = _best_chunk_for(chunks, r["file_name"], r["page"], r["heading"])
            if hit and isinstance(hit.get("bbox"), list):
                r["bbox"] = hit["bbox"]
                if hit.get("page_height_pt") is not None:
                    r["page_height_pt"] = hit["page_height_pt"]
                if hit.get("bbox_origin"):
                    r["bbox_origin"] = hit["bbox_origin"]

    latency_ms = int((time.time() - t0) * 1000)

    # Log snippet search completion
    api_logger_instance.log_performance(
        request_id=request_id,
        operation="snippet_search_complete",
        duration=latency_ms / 1000,
        results_count=len(results),
        collection_id=collection_id
    )

    response_data = {
        "selected_text": selected_text,
        "collection_id": collection_id,
        "results": results,
        "total_results": len(results),
        "latency_ms": latency_ms
    }

    return JSONResponse(content=response_data)


# -------------------------------------------
# Find document collection by filename
# -------------------------------------------
@app.get("/docs/find/{doc_id}")
async def find_document_collection(doc_id: str):
    """
    Find which collection contains a specific document by filename.
    
    Args:
        doc_id: The document filename
        
    Returns:
        Collection ID that contains the document, or null if not found
    """
    # Prevent path traversal; keep only the base name
    safe_name = os.path.basename(doc_id)
    
    try:
        for workspace_dir in workspace_manager.base_workspace_dir.iterdir():
            if workspace_dir.is_dir() and workspace_dir.name.startswith('col_'):
                uploads_dir = workspace_dir / "uploads"
                if uploads_dir.exists():
                    pdf_path = uploads_dir / safe_name
                    if pdf_path.exists():
                        return {"collection_id": workspace_dir.name, "filename": safe_name}
    except Exception:
        pass
    
    # Check legacy uploads directory
    pdf_path = INPUT_DIR / safe_name
    if pdf_path.exists():
        return {"collection_id": None, "filename": safe_name}
    
    raise HTTPException(status_code=404, detail=f"Document not found: {safe_name}")


# -------------------------------------------
# Serve original PDFs INLINE for the viewer
# -------------------------------------------
@app.get("/docs/{doc_id}")
async def get_document(doc_id: str, collection_id: Optional[str] = None):
    """
    Streams the raw PDF by filename from uploads/, forcing inline display so
    browsers and the Adobe viewer render it instead of downloading.
    
    Args:
        doc_id: The document filename
        collection_id: Optional collection ID to look in specific workspace
    """
    # Prevent path traversal; keep only the base name
    safe_name = os.path.basename(doc_id)
    
    # Try workspace-based path first if collection_id is provided
    if collection_id and collection_id.startswith('col_'):
        try:
            uploads_dir = workspace_manager.get_uploads_path(collection_id)
            pdf_path = uploads_dir / safe_name
            if pdf_path.exists():
                headers = {"Content-Disposition": f'inline; filename="{safe_name}"'}
                return FileResponse(path=pdf_path, media_type="application/pdf", headers=headers)
        except Exception:
            pass  # Fall back to search
    
    # If no collection_id provided or not found, search through all workspaces
    if not collection_id:
        try:
            for workspace_dir in workspace_manager.base_workspace_dir.iterdir():
                if workspace_dir.is_dir() and workspace_dir.name.startswith('col_'):
                    uploads_dir = workspace_dir / "uploads"
                    if uploads_dir.exists():
                        pdf_path = uploads_dir / safe_name
                        if pdf_path.exists():
                            headers = {"Content-Disposition": f'inline; filename="{safe_name}"'}
                            return FileResponse(path=pdf_path, media_type="application/pdf", headers=headers)
        except Exception:
            pass  # Fall back to legacy path
    
    # Fall back to legacy uploads directory
    pdf_path = INPUT_DIR / safe_name
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {safe_name}")

    # Force inline display (avoid Save dialog)
    headers = {"Content-Disposition": f'inline; filename="{safe_name}"'}
    return FileResponse(path=pdf_path, media_type="application/pdf", headers=headers)


@app.get("/healthz")
async def health_check():
    return {"status": "ok"}

@app.get("/concurrency/status")
async def get_concurrency_status_endpoint():
    """
    Get current concurrency status for all services.
    
    Returns:
        Current concurrency metrics for all services
    """
    return get_concurrency_status()

@app.get("/concurrency/status/{service_name}")
async def get_service_concurrency_status(service_name: str):
    """
    Get concurrency status for a specific service.
    
    Args:
        service_name: Name of the service to check
        
    Returns:
        Concurrency metrics for the specified service
    """
    return get_concurrency_status(service_name)


@app.post("/cleanall")
async def clean_all_workspaces(request: Request):
    """
    Clean up all workspaces (uploads, vector stores, summaries).
    This removes all collections and their associated data.
    """
    request_id = getattr(request.state, 'request_id', get_request_id())
    
    try:
        # Clean up all workspaces
        cleaned_count = workspace_manager.cleanup_all_workspaces()
        
        # Also clean up legacy directories for backward compatibility
        legacy_cleaned = 0
        
        # Clean legacy uploads directory
        if INPUT_DIR.exists():
            _clear_dir(INPUT_DIR)
            legacy_cleaned += 1
        
        # Clean legacy vector store collections
        if VECTOR_DIR.exists():
            _clear_collection_folders()
            legacy_cleaned += 1
        
        # Clean legacy summaries directory
        legacy_summaries_dir = Path("collection_summary_service/summaries")
        if legacy_summaries_dir.exists():
            _clear_dir(legacy_summaries_dir)
            legacy_cleaned += 1
        
        # Log cleanup operation
        api_logger_instance.log_performance(
            request_id=request_id,
            operation="clean_all_workspaces",
            duration=0,
            workspaces_cleaned=cleaned_count,
            legacy_dirs_cleaned=legacy_cleaned
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"Cleaned up {cleaned_count} workspaces and {legacy_cleaned} legacy directories",
                "workspaces_cleaned": cleaned_count,
                "legacy_dirs_cleaned": legacy_cleaned
            }
        )
        
    except Exception as e:
        api_logger_instance.log_performance(
            request_id=request_id,
            operation="clean_all_workspaces_failed",
            duration=0,
            error=str(e)
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Failed to clean workspaces: {str(e)}"
            }
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "title": "Document Insight & Engagement System",
        "version": "1.0.0",
        "description": "Enhanced PDF retrieval system with insights, chat, podcast, and relevance analysis",
        "services": {
            "core": {
                "collections": "Build and manage document collections",
                "query": "Query documents with persona-based search",
                "snippets": "Get relevant snippets from selected text",
                "docs": "Serve PDF documents"
            },
            "insights": "Generate contextual insights from document content",
            "chat": "Interactive chat with LLM using document context",
            "podcast": "Generate audio podcasts from document content",
            "relevance": "Analyze and explain relevance between texts",
            "collection_summary": "Generate comprehensive summaries of entire document collections"
        },
        "endpoints": {
            "core": ["/collections/build", "/query", "/snippets", "/docs/{doc_id}"],
            "insights": ["/insights/generate", "/insights/generate_quick"],
            "chat": ["/chat/sessions", "/chat/sessions/{id}/chat"],
            "podcast": ["/podcast/generate_from_snippets", "/podcast/generate_from_text", "/podcast/generate_from_collection"],
            "relevance": ["/relevance/analyze_single", "/relevance/analyze_multiple"],
            "collection_summary": ["/collection_summary/generate", "/collection_summary/get/{id}"]
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
