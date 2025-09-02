"""
Performance monitoring for podcast generation service.
"""

import time
import asyncio
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class PodcastPerformanceMonitor:
    """Monitor podcast generation performance metrics."""
    
    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self.metrics = defaultdict(lambda: {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'concurrent_requests': 0,
            'max_concurrent_requests': 0,
            'recent_processing_times': deque(maxlen=100)
        })
        self.lock = threading.Lock()
        self.running = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start the performance monitoring thread."""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Podcast performance monitoring started")
        
    def stop_monitoring(self):
        """Stop the performance monitoring thread."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Podcast performance monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._log_performance_metrics()
                time.sleep(self.log_interval)
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                
    def _log_performance_metrics(self):
        """Log current performance metrics."""
        with self.lock:
            for podcast_type, metrics in self.metrics.items():
                if metrics['total_requests'] > 0:
                    avg_time = metrics['total_processing_time'] / metrics['successful_requests']
                    cache_hit_rate = (metrics['cache_hits'] / (metrics['cache_hits'] + metrics['cache_misses'])) * 100 if (metrics['cache_hits'] + metrics['cache_misses']) > 0 else 0
                    
                    logger.info(f"Podcast Performance - {podcast_type}: "
                              f"Requests: {metrics['total_requests']}, "
                              f"Success: {metrics['successful_requests']}, "
                              f"Failed: {metrics['failed_requests']}, "
                              f"Avg Time: {avg_time:.2f}s, "
                              f"Cache Hit Rate: {cache_hit_rate:.1f}%, "
                              f"Max Concurrent: {metrics['max_concurrent_requests']}")
                    
    def record_request_start(self, podcast_type: str):
        """Record the start of a podcast generation request."""
        with self.lock:
            self.metrics[podcast_type]['total_requests'] += 1
            self.metrics[podcast_type]['concurrent_requests'] += 1
            current_concurrent = self.metrics[podcast_type]['concurrent_requests']
            if current_concurrent > self.metrics[podcast_type]['max_concurrent_requests']:
                self.metrics[podcast_type]['max_concurrent_requests'] = current_concurrent
                
    def record_request_complete(self, podcast_type: str, processing_time: float, success: bool, cache_hit: bool):
        """Record the completion of a podcast generation request."""
        with self.lock:
            self.metrics[podcast_type]['concurrent_requests'] -= 1
            
            if success:
                self.metrics[podcast_type]['successful_requests'] += 1
                self.metrics[podcast_type]['total_processing_time'] += processing_time
                self.metrics[podcast_type]['recent_processing_times'].append(processing_time)
            else:
                self.metrics[podcast_type]['failed_requests'] += 1
                
            if cache_hit:
                self.metrics[podcast_type]['cache_hits'] += 1
            else:
                self.metrics[podcast_type]['cache_misses'] += 1
                
    def get_performance_summary(self) -> Dict:
        """Get a summary of current performance metrics."""
        with self.lock:
            summary = {}
            for podcast_type, metrics in self.metrics.items():
                if metrics['total_requests'] > 0:
                    summary[podcast_type] = {
                        'total_requests': metrics['total_requests'],
                        'success_rate': (metrics['successful_requests'] / metrics['total_requests']) * 100,
                        'avg_processing_time': metrics['total_processing_time'] / metrics['successful_requests'] if metrics['successful_requests'] > 0 else 0,
                        'cache_hit_rate': (metrics['cache_hits'] / (metrics['cache_hits'] + metrics['cache_misses'])) * 100 if (metrics['cache_hits'] + metrics['cache_misses']) > 0 else 0,
                        'current_concurrent_requests': metrics['concurrent_requests'],
                        'max_concurrent_requests': metrics['max_concurrent_requests']
                    }
            return summary
            
    def export_metrics(self, file_path: str):
        """Export metrics to a JSON file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.get_performance_summary(), f, indent=2)
            logger.info(f"Performance metrics exported to {file_path}")
        except Exception as e:
            logger.error(f"Failed to export performance metrics: {e}")

# Global performance monitor instance
performance_monitor = PodcastPerformanceMonitor()

def start_performance_monitoring():
    """Start the global performance monitor."""
    performance_monitor.start_monitoring()
    
def stop_performance_monitoring():
    """Stop the global performance monitor."""
    performance_monitor.stop_monitoring()
    
def get_performance_summary():
    """Get current performance summary."""
    return performance_monitor.get_performance_summary()
