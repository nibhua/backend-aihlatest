"""
Global Concurrency Manager for AIH-VAL Backend

This module provides centralized concurrency control across all backend services
to prevent system overload and ensure optimal performance.
"""

import asyncio
import time
import threading
from typing import Dict, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import logging
import os

logger = logging.getLogger(__name__)

@dataclass
class ConcurrencyLimits:
    """Configuration for concurrency limits per service."""
    max_concurrent: int
    max_queue_size: int
    timeout_seconds: int
    priority: int  # Higher number = higher priority

class ConcurrencyManager:
    """Global concurrency manager for all backend services."""
    
    def __init__(self):
        # Default concurrency limits - can be overridden via environment variables
        self.limits = {
            'podcast_generation': ConcurrencyLimits(
                max_concurrent=int(os.getenv("MAX_CONCURRENT_PODCASTS", "3")),
                max_queue_size=int(os.getenv("PODCAST_QUEUE_SIZE", "10")),
                timeout_seconds=int(os.getenv("PODCAST_TIMEOUT", "300")),
                priority=3
            ),
            'query_processing': ConcurrencyLimits(
                max_concurrent=int(os.getenv("MAX_CONCURRENT_QUERIES", "10")),
                max_queue_size=int(os.getenv("QUERY_QUEUE_SIZE", "50")),
                timeout_seconds=int(os.getenv("QUERY_TIMEOUT", "60")),
                priority=5
            ),
            'collection_build': ConcurrencyLimits(
                max_concurrent=int(os.getenv("MAX_CONCURRENT_BUILDS", "2")),
                max_queue_size=int(os.getenv("BUILD_QUEUE_SIZE", "5")),
                timeout_seconds=int(os.getenv("BUILD_TIMEOUT", "600")),
                priority=4
            ),
            'snippet_processing': ConcurrencyLimits(
                max_concurrent=int(os.getenv("MAX_CONCURRENT_SNIPPETS", "8")),
                max_queue_size=int(os.getenv("SNIPPET_QUEUE_SIZE", "30")),
                timeout_seconds=int(os.getenv("SNIPPET_TIMEOUT", "30")),
                priority=4
            ),
            'chat_processing': ConcurrencyLimits(
                max_concurrent=int(os.getenv("MAX_CONCURRENT_CHATS", "5")),
                max_queue_size=int(os.getenv("CHAT_QUEUE_SIZE", "20")),
                timeout_seconds=int(os.getenv("CHAT_TIMEOUT", "120")),
                priority=3
            ),
            'summary_generation': ConcurrencyLimits(
                max_concurrent=int(os.getenv("MAX_CONCURRENT_SUMMARIES", "2")),
                max_queue_size=int(os.getenv("SUMMARY_QUEUE_SIZE", "10")),
                timeout_seconds=int(os.getenv("SUMMARY_TIMEOUT", "300")),
                priority=2
            )
        }
        
        # Semaphores for each service
        self.semaphores: Dict[str, asyncio.Semaphore] = {}
        self.queues: Dict[str, deque] = {}
        self.active_requests: Dict[str, int] = defaultdict(int)
        self.request_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'queued_requests': 0,
            'timeout_requests': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0,
            'max_concurrent_reached': 0,
            'current_queue_size': 0
        })
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Initialize semaphores
        self._initialize_semaphores()
        
        # Start monitoring thread
        self.monitoring_enabled = os.getenv("ENABLE_CONCURRENCY_MONITORING", "true").lower() == "true"
        if self.monitoring_enabled:
            self._start_monitoring()
    
    def _initialize_semaphores(self):
        """Initialize semaphores for each service."""
        for service_name, limits in self.limits.items():
            self.semaphores[service_name] = asyncio.Semaphore(limits.max_concurrent)
            self.queues[service_name] = deque(maxlen=limits.max_queue_size)
    
    def _start_monitoring(self):
        """Start the monitoring thread."""
        def monitor_loop():
            while True:
                try:
                    self._log_metrics()
                    time.sleep(int(os.getenv("CONCURRENCY_MONITOR_INTERVAL", "30")))
                except Exception as e:
                    logger.error(f"Error in concurrency monitoring: {e}")
                    time.sleep(10)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info("Concurrency monitoring started")
    
    def _log_metrics(self):
        """Log current concurrency metrics."""
        with self.lock:
            for service_name, metrics in self.request_metrics.items():
                if metrics['total_requests'] > 0:
                    success_rate = (metrics['successful_requests'] / metrics['total_requests']) * 100
                    logger.info(f"Concurrency Metrics - {service_name}: "
                              f"Active: {self.active_requests[service_name]}, "
                              f"Queue: {len(self.queues[service_name])}, "
                              f"Total: {metrics['total_requests']}, "
                              f"Success: {success_rate:.1f}%, "
                              f"Avg Time: {metrics['avg_processing_time']:.2f}s")
    
    async def acquire_slot(self, service_name: str, request_id: str = None) -> bool:
        """
        Acquire a concurrency slot for a service.
        
        Args:
            service_name: Name of the service requesting the slot
            request_id: Optional request identifier for tracking
            
        Returns:
            True if slot acquired successfully, False if queue is full
        """
        if service_name not in self.semaphores:
            logger.warning(f"Unknown service: {service_name}")
            return True  # Allow unknown services to proceed
        
        with self.lock:
            self.request_metrics[service_name]['total_requests'] += 1
            
            # Check if queue is full
            if len(self.queues[service_name]) >= self.limits[service_name].max_queue_size:
                self.request_metrics[service_name]['queued_requests'] += 1
                logger.warning(f"Queue full for {service_name}, rejecting request")
                return False
            
            # Add to queue
            self.queues[service_name].append({
                'request_id': request_id,
                'timestamp': time.time(),
                'service': service_name
            })
            self.request_metrics[service_name]['current_queue_size'] = len(self.queues[service_name])
        
        try:
            # Wait for semaphore with timeout
            await asyncio.wait_for(
                self.semaphores[service_name].acquire(),
                timeout=self.limits[service_name].timeout_seconds
            )
            
            with self.lock:
                # Remove from queue
                if self.queues[service_name]:
                    self.queues[service_name].popleft()
                
                # Update active requests
                self.active_requests[service_name] += 1
                current_active = self.active_requests[service_name]
                
                # Track max concurrent reached
                if current_active > self.request_metrics[service_name]['max_concurrent_reached']:
                    self.request_metrics[service_name]['max_concurrent_reached'] = current_active
                
                self.request_metrics[service_name]['current_queue_size'] = len(self.queues[service_name])
            
            logger.debug(f"Acquired slot for {service_name}, active: {self.active_requests[service_name]}")
            return True
            
        except asyncio.TimeoutError:
            with self.lock:
                self.request_metrics[service_name]['timeout_requests'] += 1
                # Remove from queue if still there
                if self.queues[service_name]:
                    self.queues[service_name].popleft()
                self.request_metrics[service_name]['current_queue_size'] = len(self.queues[service_name])
            
            logger.warning(f"Timeout acquiring slot for {service_name}")
            return False
    
    def release_slot(self, service_name: str, processing_time: float = 0.0, success: bool = True):
        """
        Release a concurrency slot for a service.
        
        Args:
            service_name: Name of the service releasing the slot
            processing_time: Time taken to process the request
            success: Whether the request was successful
        """
        if service_name not in self.semaphores:
            return
        
        with self.lock:
            # Update active requests
            if self.active_requests[service_name] > 0:
                self.active_requests[service_name] -= 1
            
            # Update metrics
            if success:
                self.request_metrics[service_name]['successful_requests'] += 1
            else:
                self.request_metrics[service_name]['failed_requests'] += 1
            
            if processing_time > 0:
                self.request_metrics[service_name]['total_processing_time'] += processing_time
                total_successful = self.request_metrics[service_name]['successful_requests']
                if total_successful > 0:
                    self.request_metrics[service_name]['avg_processing_time'] = (
                        self.request_metrics[service_name]['total_processing_time'] / total_successful
                    )
        
        # Release semaphore
        self.semaphores[service_name].release()
        logger.debug(f"Released slot for {service_name}, active: {self.active_requests[service_name]}")
    
    def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get current status for a specific service."""
        if service_name not in self.limits:
            return {}
        
        with self.lock:
            limits = self.limits[service_name]
            metrics = self.request_metrics[service_name]
            
            return {
                'service_name': service_name,
                'active_requests': self.active_requests[service_name],
                'max_concurrent': limits.max_concurrent,
                'queue_size': len(self.queues[service_name]),
                'max_queue_size': limits.max_queue_size,
                'utilization_percent': (self.active_requests[service_name] / limits.max_concurrent) * 100,
                'queue_utilization_percent': (len(self.queues[service_name]) / limits.max_queue_size) * 100,
                'total_requests': metrics['total_requests'],
                'success_rate': (metrics['successful_requests'] / metrics['total_requests']) * 100 if metrics['total_requests'] > 0 else 0,
                'avg_processing_time': metrics['avg_processing_time'],
                'timeout_rate': (metrics['timeout_requests'] / metrics['total_requests']) * 100 if metrics['total_requests'] > 0 else 0
            }
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all services."""
        return {service: self.get_service_status(service) for service in self.limits.keys()}
    
    def update_limits(self, service_name: str, new_limits: ConcurrencyLimits):
        """Update concurrency limits for a service (requires restart for full effect)."""
        if service_name in self.limits:
            self.limits[service_name] = new_limits
            logger.info(f"Updated limits for {service_name}: {new_limits}")
    
    def get_queue_info(self, service_name: str) -> Dict[str, Any]:
        """Get detailed queue information for a service."""
        if service_name not in self.queues:
            return {}
        
        with self.lock:
            queue = list(self.queues[service_name])
            return {
                'service_name': service_name,
                'queue_size': len(queue),
                'max_queue_size': self.limits[service_name].max_queue_size,
                'oldest_request_age': time.time() - queue[0]['timestamp'] if queue else 0,
                'queue_items': [
                    {
                        'request_id': item['request_id'],
                        'age_seconds': time.time() - item['timestamp']
                    }
                    for item in queue
                ]
            }

# Global concurrency manager instance
concurrency_manager = ConcurrencyManager()

# Convenience functions
async def acquire_concurrency_slot(service_name: str, request_id: str = None) -> bool:
    """Acquire a concurrency slot for a service."""
    return await concurrency_manager.acquire_slot(service_name, request_id)

def release_concurrency_slot(service_name: str, processing_time: float = 0.0, success: bool = True):
    """Release a concurrency slot for a service."""
    concurrency_manager.release_slot(service_name, processing_time, success)

def get_concurrency_status(service_name: str = None) -> Dict[str, Any]:
    """Get concurrency status for a service or all services."""
    if service_name:
        return concurrency_manager.get_service_status(service_name)
    return concurrency_manager.get_all_status()

# Context manager for automatic slot management
class ConcurrencySlot:
    """Context manager for automatic concurrency slot management."""
    
    def __init__(self, service_name: str, request_id: str = None):
        self.service_name = service_name
        self.request_id = request_id
        self.acquired = False
        self.start_time = None
    
    async def __aenter__(self):
        self.acquired = await acquire_concurrency_slot(self.service_name, self.request_id)
        if not self.acquired:
            raise RuntimeError(f"Failed to acquire concurrency slot for {self.service_name}")
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.acquired:
            processing_time = time.time() - self.start_time if self.start_time else 0.0
            success = exc_type is None
            release_concurrency_slot(self.service_name, processing_time, success)
