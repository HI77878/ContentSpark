#!/usr/bin/env python3
"""
Real-time monitoring dashboard for TikTok mass processing system
"""

from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import psutil
import GPUtil
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any
import os
import sys

# Add parent directory to path
sys.path.append('/home/user/tiktok_production')

from mass_processing.queue_manager import QueueManager
from utils.supabase_client import supabase

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dashboard HTML template
DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>TikTok Mass Processing Monitor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f0f2f5;
        }
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #1a73e8;
            margin: 10px 0;
        }
        .metric-label {
            color: #5f6368;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .status-badge {
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.85em;
            font-weight: 500;
        }
        .status-idle { background: #e8f5e9; color: #2e7d32; }
        .status-busy { background: #fff3e0; color: #ef6c00; }
        .status-offline { background: #ffebee; color: #c62828; }
        .chart-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .worker-card {
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            border: 1px solid #e0e0e0;
        }
        .queue-item {
            background: #f8f9fa;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 5px;
            font-size: 0.9em;
        }
        .progress-bar-custom {
            height: 8px;
            border-radius: 4px;
            background: #e0e0e0;
            overflow: hidden;
            margin: 5px 0;
        }
        .progress-fill {
            height: 100%;
            background: #1a73e8;
            transition: width 0.3s ease;
        }
        .alert-banner {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            padding: 15px;
            background: #d32f2f;
            color: white;
            text-align: center;
            display: none;
            z-index: 1000;
        }
    </style>
</head>
<body>
    <div class="alert-banner" id="alertBanner"></div>
    
    <div class="container-fluid py-4">
        <h1 class="mb-4">🎬 TikTok Mass Processing Monitor</h1>
        
        <!-- Key Metrics -->
        <div class="row">
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-label">Total Processed</div>
                    <div class="metric-value" id="totalProcessed">-</div>
                    <small class="text-muted">Videos analyzed</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-label">Queue Size</div>
                    <div class="metric-value" id="queueSize">-</div>
                    <small class="text-muted">Pending videos</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-label">Throughput</div>
                    <div class="metric-value" id="throughput">-</div>
                    <small class="text-muted">Videos/hour</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-label">GPU Utilization</div>
                    <div class="metric-value" id="gpuUtil">-</div>
                    <small class="text-muted">Average %</small>
                </div>
            </div>
        </div>
        
        <!-- Charts -->
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="chart-container">
                    <h5>Processing Throughput</h5>
                    <canvas id="throughputChart" height="100"></canvas>
                </div>
            </div>
            <div class="col-md-6">
                <div class="chart-container">
                    <h5>GPU Utilization</h5>
                    <canvas id="gpuChart" height="100"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Queue Details -->
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="chart-container">
                    <h5>Queue Status</h5>
                    <div id="queueDetails"></div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="chart-container">
                    <h5>Processing Times</h5>
                    <canvas id="processingTimeChart" height="150"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Worker Status -->
        <div class="mt-4">
            <h3>Worker Status</h3>
            <div class="row" id="workerStatus"></div>
        </div>
        
        <!-- Recent Errors -->
        <div class="mt-4">
            <h3>Recent Activity</h3>
            <div class="chart-container">
                <div id="recentActivity"></div>
            </div>
        </div>
    </div>
    
    <script>
        // Chart configurations
        const chartOptions = {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: { beginAtZero: true }
            }
        };
        
        // Initialize charts
        const throughputCtx = document.getElementById('throughputChart').getContext('2d');
        const throughputChart = new Chart(throughputCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Videos/Hour',
                    data: [],
                    borderColor: '#1a73e8',
                    backgroundColor: 'rgba(26, 115, 232, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: chartOptions
        });
        
        const gpuCtx = document.getElementById('gpuChart').getContext('2d');
        const gpuChart = new Chart(gpuCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'GPU %',
                    data: [],
                    borderColor: '#34a853',
                    backgroundColor: 'rgba(52, 168, 83, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                ...chartOptions,
                scales: {
                    y: { beginAtZero: true, max: 100 }
                }
            }
        });
        
        const processingTimeCtx = document.getElementById('processingTimeChart').getContext('2d');
        const processingTimeChart = new Chart(processingTimeCtx, {
            type: 'bar',
            data: {
                labels: ['Download', 'GPU Processing', 'Total'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ['#4285f4', '#34a853', '#fbbc04']
                }]
            },
            options: {
                ...chartOptions,
                scales: {
                    y: { 
                        beginAtZero: true,
                        title: { display: true, text: 'Seconds' }
                    }
                }
            }
        });
        
        // Update functions
        function updateMetrics(data) {
            $('#totalProcessed').text(data.total_processed.toLocaleString());
            $('#queueSize').text(data.queue_size.toLocaleString());
            $('#throughput').text(data.throughput);
            $('#gpuUtil').text(data.gpu_utilization + '%');
            
            // Update charts
            const now = new Date().toLocaleTimeString();
            
            // Throughput chart
            throughputChart.data.labels.push(now);
            throughputChart.data.datasets[0].data.push(data.throughput);
            if (throughputChart.data.labels.length > 30) {
                throughputChart.data.labels.shift();
                throughputChart.data.datasets[0].data.shift();
            }
            throughputChart.update('none');
            
            // GPU chart
            gpuChart.data.labels.push(now);
            gpuChart.data.datasets[0].data.push(data.gpu_utilization);
            if (gpuChart.data.labels.length > 30) {
                gpuChart.data.labels.shift();
                gpuChart.data.datasets[0].data.shift();
            }
            gpuChart.update('none');
            
            // Processing times
            if (data.processing_times) {
                processingTimeChart.data.datasets[0].data = [
                    data.processing_times.avg_download_time || 0,
                    data.processing_times.avg_processing_time || 0,
                    data.processing_times.avg_total_time || 0
                ];
                processingTimeChart.update('none');
            }
        }
        
        function updateQueueDetails(data) {
            let html = '';
            const queues = data.queue_breakdown || {};
            
            for (const [name, count] of Object.entries(queues)) {
                const percentage = data.queue_size > 0 ? (count / data.queue_size * 100).toFixed(1) : 0;
                html += `
                    <div class="mb-3">
                        <div class="d-flex justify-content-between">
                            <span>${name.replace('queue_', '').toUpperCase()}</span>
                            <span>${count}</span>
                        </div>
                        <div class="progress-bar-custom">
                            <div class="progress-fill" style="width: ${percentage}%"></div>
                        </div>
                    </div>
                `;
            }
            
            $('#queueDetails').html(html);
        }
        
        function updateWorkers(workers) {
            let html = '';
            
            workers.forEach(worker => {
                const statusClass = `status-${worker.status}`;
                const taskInfo = worker.current_task ? `<br><small>${worker.current_task}</small>` : '';
                
                html += `
                    <div class="col-md-6 col-lg-4">
                        <div class="worker-card">
                            <div class="d-flex justify-content-between align-items-start">
                                <div>
                                    <strong>${worker.id}</strong>
                                    <span class="status-badge ${statusClass}">${worker.status}</span>
                                    ${taskInfo}
                                </div>
                            </div>
                            <div class="mt-2">
                                <small>GPU: ${worker.gpu_utilization}% | Mem: ${worker.memory_usage}%</small>
                                <div class="progress-bar-custom">
                                    <div class="progress-fill" style="width: ${worker.gpu_utilization}%"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            $('#workerStatus').html(html || '<p class="text-muted">No workers active</p>');
        }
        
        function updateActivity(data) {
            let html = '';
            
            if (data.recent_completions) {
                html += '<h6>Recently Completed</h6>';
                data.recent_completions.forEach(item => {
                    html += `
                        <div class="queue-item">
                            <strong>${item.tiktok_id}</strong> - 
                            ${item.successful_analyzers} analyzers - 
                            ${item.total_processing_time.toFixed(1)}s
                            <small class="text-muted float-end">${new Date(item.analyzed_at).toLocaleTimeString()}</small>
                        </div>
                    `;
                });
            }
            
            if (data.recent_errors && data.recent_errors.length > 0) {
                html += '<h6 class="mt-3 text-danger">Recent Errors</h6>';
                data.recent_errors.forEach(error => {
                    html += `
                        <div class="queue-item border-danger">
                            <strong>${error.url}</strong><br>
                            <small class="text-danger">${error.error}</small>
                        </div>
                    `;
                });
            }
            
            $('#recentActivity').html(html || '<p class="text-muted">No recent activity</p>');
        }
        
        function showAlert(message) {
            $('#alertBanner').text(message).slideDown();
            setTimeout(() => {
                $('#alertBanner').slideUp();
            }, 5000);
        }
        
        // Main update function
        function updateDashboard() {
            $.get('/api/metrics')
                .done(data => {
                    updateMetrics(data);
                    updateQueueDetails(data);
                    updateWorkers(data.workers || []);
                    updateActivity(data);
                    
                    // Check for alerts
                    if (data.alerts && data.alerts.length > 0) {
                        showAlert(data.alerts[0]);
                    }
                })
                .fail(error => {
                    console.error('Failed to fetch metrics:', error);
                    showAlert('Failed to connect to monitoring service');
                });
        }
        
        // Initial load and periodic updates
        updateDashboard();
        setInterval(updateDashboard, 2000);  // Update every 2 seconds
        
        // Auto-refresh page every hour to prevent memory leaks
        setTimeout(() => {
            location.reload();
        }, 3600000);
    </script>
</body>
</html>
'''

@app.route('/')
def dashboard():
    """Serve monitoring dashboard"""
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/api/metrics')
def get_metrics():
    """Get current system metrics"""
    try:
        # Initialize queue manager
        queue_manager = QueueManager()
        queue_stats = queue_manager.get_queue_stats()
        
        # Get worker status from Supabase
        workers_result = supabase.table('worker_status').select('*').execute()
        workers = workers_result.data if workers_result else []
        
        # Calculate active workers
        active_workers = sum(1 for w in workers if w.get('status') == 'busy')
        
        # Get GPU stats
        gpus = GPUtil.getGPUs()
        if gpus:
            avg_gpu_util = sum(gpu.load * 100 for gpu in gpus) / len(gpus)
            avg_gpu_memory = sum(gpu.memoryUtil * 100 for gpu in gpus) / len(gpus)
        else:
            avg_gpu_util = 0
            avg_gpu_memory = 0
        
        # Calculate throughput (videos in last hour)
        hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
        processed_result = supabase.table('video_analysis').select('id').gte('analyzed_at', hour_ago).execute()
        throughput = len(processed_result.data) if processed_result else 0
        
        # Get recent completions
        recent_result = supabase.table('video_analysis').select(
            'tiktok_id', 'successful_analyzers', 'total_processing_time', 'analyzed_at'
        ).order('analyzed_at', desc=True).limit(5).execute()
        recent_completions = recent_result.data if recent_result else []
        
        # Get recent errors
        failed_result = supabase.table('processing_queue').select(
            'tiktok_url', 'error_message'
        ).eq('status', 'failed').order('completed_at', desc=True).limit(5).execute()
        recent_errors = [{'url': f['tiktok_url'], 'error': f['error_message']} 
                        for f in (failed_result.data if failed_result else [])]
        
        # Get processing time stats
        processing_times = queue_manager.get_processing_time_stats()
        
        # Check for alerts
        alerts = []
        if avg_gpu_util < 70 and active_workers > 0:
            alerts.append(f"Low GPU utilization: {avg_gpu_util:.1f}%")
        if queue_stats.get('queue_failed', 0) > 10:
            alerts.append(f"High failure rate: {queue_stats['queue_failed']} failed tasks")
        
        return jsonify({
            'total_processed': queue_stats.get('processed', 0),
            'queue_size': queue_stats.get('total_pending', 0),
            'active_workers': active_workers,
            'gpu_utilization': round(avg_gpu_util, 1),
            'gpu_memory': round(avg_gpu_memory, 1),
            'throughput': throughput,
            'workers': workers,
            'queue_breakdown': {k: v for k, v in queue_stats.items() if k.startswith('queue_')},
            'processing_times': processing_times,
            'recent_completions': recent_completions,
            'recent_errors': recent_errors,
            'alerts': alerts,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/queue/<action>', methods=['POST'])
def manage_queue(action):
    """Queue management endpoints"""
    try:
        queue_manager = QueueManager()
        
        if action == 'add':
            data = request.json
            urls = data.get('urls', [])
            priority = data.get('priority', 5)
            
            if isinstance(urls, str):
                urls = [urls]
                
            results = queue_manager.add_urls_batch(urls, priority)
            return jsonify({
                'success': True,
                'added': sum(1 for v in results.values() if v),
                'results': results
            })
            
        elif action == 'requeue':
            requeued = queue_manager.requeue_failed()
            return jsonify({
                'success': True,
                'requeued': requeued
            })
            
        elif action == 'clear':
            # Require confirmation
            if request.json.get('confirm') == 'yes':
                queue_manager.clear_all_queues()
                return jsonify({'success': True, 'message': 'All queues cleared'})
            else:
                return jsonify({'success': False, 'message': 'Confirmation required'}), 400
                
    except Exception as e:
        logger.error(f"Queue management error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/<format>')
def export_data(format):
    """Export monitoring data"""
    try:
        queue_manager = QueueManager()
        
        if format == 'json':
            # Export comprehensive stats
            stats = {
                'timestamp': datetime.now().isoformat(),
                'queue_stats': queue_manager.get_queue_stats(),
                'processing_times': queue_manager.get_processing_time_stats(),
                'system_info': {
                    'cpu_count': psutil.cpu_count(),
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent,
                    'gpu_count': len(GPUtil.getGPUs())
                }
            }
            
            return jsonify(stats)
            
        elif format == 'csv':
            # Export recent completions as CSV
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['tiktok_id', 'username', 'duration', 'analyzers', 'processing_time', 'analyzed_at'])
            
            # Get data
            result = supabase.table('video_analysis').select('*').order('analyzed_at', desc=True).limit(1000).execute()
            
            for row in (result.data if result else []):
                writer.writerow([
                    row.get('tiktok_id'),
                    row.get('username'),
                    row.get('duration_seconds'),
                    row.get('successful_analyzers'),
                    row.get('total_processing_time'),
                    row.get('analyzed_at')
                ])
                
            output.seek(0)
            return output.getvalue(), 200, {
                'Content-Type': 'text/csv',
                'Content-Disposition': 'attachment; filename=tiktok_analysis_export.csv'
            }
            
    except Exception as e:
        logger.error(f"Export error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)