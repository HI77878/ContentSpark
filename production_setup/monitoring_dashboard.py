#!/usr/bin/env python3
"""
Real-time Monitoring Dashboard for TikTok Video Analyzer
Provides live metrics, performance tracking, and alerts
"""

import asyncio
import json
import time
import psutil
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import subprocess

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import nvidia_ml_py as nvml

# Initialize NVML for GPU monitoring
nvml.nvmlInit()

app = FastAPI(title="TikTok Analyzer Monitoring Dashboard")

# Metrics storage
class MetricsCollector:
    def __init__(self, max_history=1000):
        self.gpu_metrics = deque(maxlen=max_history)
        self.cpu_metrics = deque(maxlen=max_history)
        self.analyzer_metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.error_log = deque(maxlen=100)
        self.processing_times = deque(maxlen=max_history)
        self.active_jobs = {}
        self.db_path = "/home/user/tiktok_production/monitoring.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for persistent metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyzer_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                analyzer_name TEXT,
                success_count INTEGER,
                error_count INTEGER,
                avg_processing_time REAL,
                duplicate_rate REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                gpu_usage REAL,
                gpu_memory_used REAL,
                gpu_memory_total REAL,
                cpu_usage REAL,
                ram_usage REAL,
                disk_usage REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def collect_gpu_metrics(self) -> Dict:
        """Collect current GPU metrics"""
        try:
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            
            # GPU utilization
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Memory info
            mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Temperature
            temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
            
            # Power usage
            power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            
            metrics = {
                'timestamp': time.time(),
                'gpu_usage': util.gpu,
                'memory_used_gb': mem_info.used / (1024**3),
                'memory_total_gb': mem_info.total / (1024**3),
                'memory_percent': (mem_info.used / mem_info.total) * 100,
                'temperature': temp,
                'power_watts': power
            }
            
            self.gpu_metrics.append(metrics)
            return metrics
            
        except Exception as e:
            self.error_log.append({
                'timestamp': time.time(),
                'error': f"GPU metrics collection failed: {str(e)}"
            })
            return {}
    
    def collect_cpu_metrics(self) -> Dict:
        """Collect current CPU/System metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
            
            # Memory
            memory = psutil.virtual_memory()
            
            # Disk
            disk = psutil.disk_usage('/')
            
            # Network
            net_io = psutil.net_io_counters()
            
            metrics = {
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'cpu_per_core': cpu_per_core,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'memory_percent': memory.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_total_gb': disk.total / (1024**3),
                'disk_percent': disk.percent,
                'network_sent_mb': net_io.bytes_sent / (1024**2),
                'network_recv_mb': net_io.bytes_recv / (1024**2)
            }
            
            self.cpu_metrics.append(metrics)
            return metrics
            
        except Exception as e:
            self.error_log.append({
                'timestamp': time.time(),
                'error': f"CPU metrics collection failed: {str(e)}"
            })
            return {}
    
    def update_analyzer_metrics(self, analyzer_name: str, success: bool, 
                              processing_time: float, duplicate_rate: float = 0.0):
        """Update metrics for a specific analyzer"""
        self.analyzer_metrics[analyzer_name].append({
            'timestamp': time.time(),
            'success': success,
            'processing_time': processing_time,
            'duplicate_rate': duplicate_rate
        })
    
    def get_analyzer_stats(self, analyzer_name: str, window_minutes: int = 60) -> Dict:
        """Get statistics for a specific analyzer"""
        if analyzer_name not in self.analyzer_metrics:
            return {}
        
        cutoff_time = time.time() - (window_minutes * 60)
        recent_metrics = [m for m in self.analyzer_metrics[analyzer_name] 
                         if m['timestamp'] > cutoff_time]
        
        if not recent_metrics:
            return {}
        
        success_count = sum(1 for m in recent_metrics if m['success'])
        error_count = len(recent_metrics) - success_count
        avg_time = sum(m['processing_time'] for m in recent_metrics) / len(recent_metrics)
        avg_duplicate_rate = sum(m['duplicate_rate'] for m in recent_metrics) / len(recent_metrics)
        
        return {
            'analyzer_name': analyzer_name,
            'success_count': success_count,
            'error_count': error_count,
            'success_rate': (success_count / len(recent_metrics)) * 100 if recent_metrics else 0,
            'avg_processing_time': avg_time,
            'avg_duplicate_rate': avg_duplicate_rate,
            'total_runs': len(recent_metrics)
        }
    
    def save_metrics_to_db(self):
        """Save current metrics to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Save system metrics
        if self.gpu_metrics and self.cpu_metrics:
            gpu = self.gpu_metrics[-1]
            cpu = self.cpu_metrics[-1]
            
            cursor.execute('''
                INSERT INTO system_metrics 
                (gpu_usage, gpu_memory_used, gpu_memory_total, 
                 cpu_usage, ram_usage, disk_usage)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                gpu.get('gpu_usage', 0),
                gpu.get('memory_used_gb', 0),
                gpu.get('memory_total_gb', 0),
                cpu.get('cpu_percent', 0),
                cpu.get('memory_percent', 0),
                cpu.get('disk_percent', 0)
            ))
        
        # Save analyzer stats
        for analyzer_name in self.analyzer_metrics:
            stats = self.get_analyzer_stats(analyzer_name, window_minutes=5)
            if stats:
                cursor.execute('''
                    INSERT INTO analyzer_stats 
                    (analyzer_name, success_count, error_count, 
                     avg_processing_time, duplicate_rate)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    analyzer_name,
                    stats['success_count'],
                    stats['error_count'],
                    stats['avg_processing_time'],
                    stats['avg_duplicate_rate']
                ))
        
        conn.commit()
        conn.close()

# Global metrics collector
metrics = MetricsCollector()

# Background task to collect metrics
async def collect_metrics_loop():
    """Background task to continuously collect metrics"""
    while True:
        metrics.collect_gpu_metrics()
        metrics.collect_cpu_metrics()
        
        # Check API health
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8003/health", timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    # Update active analyzers count
                    metrics.active_jobs['api_status'] = 'healthy'
                    metrics.active_jobs['active_analyzers'] = health_data.get('active_analyzers', 0)
        except:
            metrics.active_jobs['api_status'] = 'unreachable'
        
        # Save to database every minute
        if int(time.time()) % 60 == 0:
            metrics.save_metrics_to_db()
        
        await asyncio.sleep(5)  # Collect every 5 seconds

# WebSocket for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Send current metrics
            data = {
                'gpu': metrics.gpu_metrics[-1] if metrics.gpu_metrics else {},
                'cpu': metrics.cpu_metrics[-1] if metrics.cpu_metrics else {},
                'active_jobs': metrics.active_jobs,
                'recent_errors': list(metrics.error_log)[-5:],
                'analyzer_stats': {
                    name: metrics.get_analyzer_stats(name, window_minutes=5)
                    for name in metrics.analyzer_metrics
                }
            }
            
            await websocket.send_json(data)
            await asyncio.sleep(2)  # Update every 2 seconds
            
    except WebSocketDisconnect:
        pass

# REST API Endpoints
@app.get("/api/metrics/current")
async def get_current_metrics():
    """Get current system metrics"""
    return JSONResponse({
        'gpu': metrics.gpu_metrics[-1] if metrics.gpu_metrics else {},
        'cpu': metrics.cpu_metrics[-1] if metrics.cpu_metrics else {},
        'timestamp': datetime.now().isoformat()
    })

@app.get("/api/metrics/history")
async def get_metrics_history(minutes: int = 60):
    """Get historical metrics"""
    cutoff_time = time.time() - (minutes * 60)
    
    return JSONResponse({
        'gpu': [m for m in metrics.gpu_metrics if m['timestamp'] > cutoff_time],
        'cpu': [m for m in metrics.cpu_metrics if m['timestamp'] > cutoff_time]
    })

@app.get("/api/analyzers/stats")
async def get_analyzer_stats():
    """Get analyzer performance statistics"""
    stats = {}
    for analyzer_name in metrics.analyzer_metrics:
        stats[analyzer_name] = metrics.get_analyzer_stats(analyzer_name)
    
    return JSONResponse(stats)

@app.get("/api/alerts")
async def get_alerts():
    """Get system alerts"""
    alerts = []
    
    # Check GPU memory
    if metrics.gpu_metrics:
        latest_gpu = metrics.gpu_metrics[-1]
        if latest_gpu.get('memory_percent', 0) > 90:
            alerts.append({
                'level': 'critical',
                'message': f"GPU memory critical: {latest_gpu['memory_percent']:.1f}%",
                'timestamp': datetime.now().isoformat()
            })
        elif latest_gpu.get('memory_percent', 0) > 80:
            alerts.append({
                'level': 'warning',
                'message': f"GPU memory high: {latest_gpu['memory_percent']:.1f}%",
                'timestamp': datetime.now().isoformat()
            })
    
    # Check API status
    if metrics.active_jobs.get('api_status') == 'unreachable':
        alerts.append({
            'level': 'critical',
            'message': "API service is unreachable",
            'timestamp': datetime.now().isoformat()
        })
    
    # Check analyzer error rates
    for analyzer_name, stats in metrics.analyzer_metrics.items():
        analyzer_stats = metrics.get_analyzer_stats(analyzer_name, window_minutes=10)
        if analyzer_stats and analyzer_stats.get('success_rate', 100) < 80:
            alerts.append({
                'level': 'warning',
                'message': f"{analyzer_name} success rate low: {analyzer_stats['success_rate']:.1f}%",
                'timestamp': datetime.now().isoformat()
            })
    
    return JSONResponse(alerts)

# HTML Dashboard
@app.get("/")
async def dashboard():
    """Serve monitoring dashboard HTML"""
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>TikTok Analyzer Monitoring</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: #e0e0e0;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            color: #888;
            font-size: 0.9em;
        }
        .chart-container {
            background: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            height: 300px;
        }
        .alert {
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .alert-warning {
            background: #ff9800;
            color: #000;
        }
        .alert-critical {
            background: #f44336;
            color: #fff;
        }
        .status-healthy {
            color: #4caf50;
        }
        .status-unhealthy {
            color: #f44336;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 TikTok Analyzer Monitoring Dashboard</h1>
        
        <div id="alerts"></div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">GPU Usage</div>
                <div class="metric-value" id="gpu-usage">--%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">GPU Memory</div>
                <div class="metric-value" id="gpu-memory">-- / -- GB</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">CPU Usage</div>
                <div class="metric-value" id="cpu-usage">--%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">API Status</div>
                <div class="metric-value" id="api-status">--</div>
            </div>
        </div>
        
        <div class="chart-container">
            <canvas id="gpu-chart"></canvas>
        </div>
        
        <div class="chart-container">
            <canvas id="analyzer-chart"></canvas>
        </div>
        
        <div id="analyzer-stats"></div>
    </div>
    
    <script>
        // WebSocket connection
        const ws = new WebSocket('ws://localhost:5000/ws');
        
        // Chart setup
        const gpuChart = new Chart(document.getElementById('gpu-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'GPU Usage %',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }, {
                    label: 'GPU Memory %',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
        
        // Update metrics
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            // Update GPU metrics
            if (data.gpu) {
                document.getElementById('gpu-usage').textContent = 
                    `${data.gpu.gpu_usage?.toFixed(1) || 0}%`;
                document.getElementById('gpu-memory').textContent = 
                    `${data.gpu.memory_used_gb?.toFixed(1) || 0} / ${data.gpu.memory_total_gb?.toFixed(1) || 0} GB`;
                
                // Update chart
                const time = new Date().toLocaleTimeString();
                gpuChart.data.labels.push(time);
                gpuChart.data.datasets[0].data.push(data.gpu.gpu_usage || 0);
                gpuChart.data.datasets[1].data.push(data.gpu.memory_percent || 0);
                
                // Keep last 50 points
                if (gpuChart.data.labels.length > 50) {
                    gpuChart.data.labels.shift();
                    gpuChart.data.datasets.forEach(d => d.data.shift());
                }
                gpuChart.update();
            }
            
            // Update CPU metrics
            if (data.cpu) {
                document.getElementById('cpu-usage').textContent = 
                    `${data.cpu.cpu_percent?.toFixed(1) || 0}%`;
            }
            
            // Update API status
            const apiStatus = data.active_jobs?.api_status || 'unknown';
            const statusEl = document.getElementById('api-status');
            statusEl.textContent = apiStatus;
            statusEl.className = `metric-value status-${apiStatus === 'healthy' ? 'healthy' : 'unhealthy'}`;
            
            // Update analyzer stats
            if (data.analyzer_stats) {
                let statsHtml = '<h2>Analyzer Performance</h2><div class="metrics-grid">';
                for (const [name, stats] of Object.entries(data.analyzer_stats)) {
                    if (stats.total_runs > 0) {
                        statsHtml += `
                            <div class="metric-card">
                                <div class="metric-label">${name}</div>
                                <div>Success Rate: ${stats.success_rate.toFixed(1)}%</div>
                                <div>Avg Time: ${stats.avg_processing_time.toFixed(1)}s</div>
                                <div>Runs: ${stats.total_runs}</div>
                            </div>
                        `;
                    }
                }
                statsHtml += '</div>';
                document.getElementById('analyzer-stats').innerHTML = statsHtml;
            }
        };
        
        // Fetch alerts
        async function fetchAlerts() {
            try {
                const response = await fetch('/api/alerts');
                const alerts = await response.json();
                
                let alertsHtml = '';
                alerts.forEach(alert => {
                    alertsHtml += `
                        <div class="alert alert-${alert.level}">
                            ${alert.message} (${new Date(alert.timestamp).toLocaleTimeString()})
                        </div>
                    `;
                });
                document.getElementById('alerts').innerHTML = alertsHtml;
            } catch (e) {
                console.error('Failed to fetch alerts:', e);
            }
        }
        
        // Update alerts every 10 seconds
        setInterval(fetchAlerts, 10000);
        fetchAlerts();
    </script>
</body>
</html>
    """)

# Start background tasks
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(collect_metrics_loop())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)