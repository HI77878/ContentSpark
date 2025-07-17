#!/usr/bin/env python3
"""
Quality Monitor for TikTok Video Analyzer
Automatically detects degradation and sends alerts
"""

import asyncio
import json
import logging
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sqlite3
import requests
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/user/tiktok_production/logs/quality_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QualityAlert:
    def __init__(self, level: str, analyzer: str, metric: str, 
                 current_value: float, threshold: float, message: str):
        self.level = level  # warning, critical
        self.analyzer = analyzer
        self.metric = metric
        self.current_value = current_value
        self.threshold = threshold
        self.message = message
        self.timestamp = datetime.now()

class QualityMonitor:
    def __init__(self):
        self.monitoring_db = "/home/user/tiktok_production/monitoring.db"
        self.results_dir = Path("/home/user/tiktok_production/results")
        self.api_url = "http://localhost:8003"
        self.alerts: List[QualityAlert] = []
        
        # Quality thresholds
        self.thresholds = {
            'duplicate_rate': {'warning': 5.0, 'critical': 10.0},
            'processing_time_factor': {'warning': 3.0, 'critical': 5.0},
            'gpu_memory_percent': {'warning': 80.0, 'critical': 90.0},
            'analyzer_success_rate': {'warning': 90.0, 'critical': 80.0},
            'missing_fields_rate': {'warning': 5.0, 'critical': 10.0}
        }
        
        # Email configuration (optional)
        self.email_config = {
            'enabled': False,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'from_email': 'monitor@example.com',
            'to_emails': ['admin@example.com'],
            'password': 'your_password'
        }
    
    async def check_analyzer_quality(self) -> List[QualityAlert]:
        """Check individual analyzer quality metrics"""
        alerts = []
        
        try:
            # Get recent analysis results
            recent_results = self._get_recent_results(hours=1)
            
            # Analyzer-specific checks
            analyzer_metrics = defaultdict(lambda: {
                'total': 0, 'duplicates': 0, 'missing_fields': 0,
                'processing_times': [], 'errors': 0
            })
            
            for result_path in recent_results:
                try:
                    with open(result_path, 'r') as f:
                        data = json.load(f)
                    
                    # Check each analyzer's results
                    for analyzer_name, results in data.get('analyzer_results', {}).items():
                        metrics = analyzer_metrics[analyzer_name]
                        metrics['total'] += 1
                        
                        # Check for duplicates (Qwen2-VL specific)
                        if analyzer_name == 'qwen2_vl_temporal' and 'segments' in results:
                            segments = results['segments']
                            descriptions = [s.get('description', '') for s in segments]
                            unique_descriptions = len(set(descriptions))
                            if segments:
                                duplicate_rate = (len(segments) - unique_descriptions) / len(segments) * 100
                                if duplicate_rate > 0:
                                    metrics['duplicates'] += 1
                                
                                # Check threshold
                                if duplicate_rate > self.thresholds['duplicate_rate']['critical']:
                                    alerts.append(QualityAlert(
                                        'critical', analyzer_name, 'duplicate_rate',
                                        duplicate_rate, self.thresholds['duplicate_rate']['critical'],
                                        f"{analyzer_name} has {duplicate_rate:.1f}% duplicate descriptions"
                                    ))
                        
                        # Check for missing fields
                        expected_fields = self._get_expected_fields(analyzer_name)
                        if expected_fields and 'segments' in results:
                            for segment in results.get('segments', []):
                                missing = [f for f in expected_fields if f not in segment]
                                if missing:
                                    metrics['missing_fields'] += 1
                                    break
                
                except Exception as e:
                    logger.error(f"Error checking {result_path}: {e}")
            
            # Calculate rates and generate alerts
            for analyzer_name, metrics in analyzer_metrics.items():
                if metrics['total'] > 0:
                    # Missing fields rate
                    missing_rate = (metrics['missing_fields'] / metrics['total']) * 100
                    if missing_rate > self.thresholds['missing_fields_rate']['warning']:
                        level = 'critical' if missing_rate > self.thresholds['missing_fields_rate']['critical'] else 'warning'
                        alerts.append(QualityAlert(
                            level, analyzer_name, 'missing_fields_rate',
                            missing_rate, self.thresholds['missing_fields_rate'][level],
                            f"{analyzer_name} missing required fields in {missing_rate:.1f}% of results"
                        ))
            
        except Exception as e:
            logger.error(f"Error in analyzer quality check: {e}")
        
        return alerts
    
    async def check_system_performance(self) -> List[QualityAlert]:
        """Check system-wide performance metrics"""
        alerts = []
        
        try:
            # Check API health
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                
                # GPU memory check
                gpu_memory = health_data.get('gpu', {})
                if gpu_memory.get('gpu_memory'):
                    memory_percent = float(gpu_memory['gpu_memory']['utilization'].rstrip('%'))
                    
                    if memory_percent > self.thresholds['gpu_memory_percent']['critical']:
                        alerts.append(QualityAlert(
                            'critical', 'system', 'gpu_memory_percent',
                            memory_percent, self.thresholds['gpu_memory_percent']['critical'],
                            f"GPU memory critical: {memory_percent:.1f}%"
                        ))
                    elif memory_percent > self.thresholds['gpu_memory_percent']['warning']:
                        alerts.append(QualityAlert(
                            'warning', 'system', 'gpu_memory_percent',
                            memory_percent, self.thresholds['gpu_memory_percent']['warning'],
                            f"GPU memory high: {memory_percent:.1f}%"
                        ))
            
            # Check processing times from monitoring DB
            if Path(self.monitoring_db).exists():
                conn = sqlite3.connect(self.monitoring_db)
                cursor = conn.cursor()
                
                # Get recent processing times
                cursor.execute('''
                    SELECT analyzer_name, AVG(avg_processing_time) as avg_time
                    FROM analyzer_stats
                    WHERE timestamp > datetime('now', '-1 hour')
                    GROUP BY analyzer_name
                ''')
                
                for analyzer_name, avg_time in cursor.fetchall():
                    if avg_time:
                        # Estimate realtime factor (assuming average 30s videos)
                        realtime_factor = avg_time / 30.0
                        
                        if realtime_factor > self.thresholds['processing_time_factor']['critical']:
                            alerts.append(QualityAlert(
                                'critical', analyzer_name, 'processing_time_factor',
                                realtime_factor, self.thresholds['processing_time_factor']['critical'],
                                f"{analyzer_name} processing too slow: {realtime_factor:.1f}x realtime"
                            ))
                
                conn.close()
            
        except Exception as e:
            logger.error(f"Error in system performance check: {e}")
            alerts.append(QualityAlert(
                'critical', 'system', 'api_health',
                0, 0, f"API health check failed: {str(e)}"
            ))
        
        return alerts
    
    async def check_analyzer_success_rates(self) -> List[QualityAlert]:
        """Check analyzer success rates"""
        alerts = []
        
        try:
            if Path(self.monitoring_db).exists():
                conn = sqlite3.connect(self.monitoring_db)
                cursor = conn.cursor()
                
                # Get success rates
                cursor.execute('''
                    SELECT analyzer_name, 
                           SUM(success_count) as successes,
                           SUM(error_count) as errors
                    FROM analyzer_stats
                    WHERE timestamp > datetime('now', '-1 hour')
                    GROUP BY analyzer_name
                ''')
                
                for analyzer_name, successes, errors in cursor.fetchall():
                    total = (successes or 0) + (errors or 0)
                    if total > 0:
                        success_rate = (successes / total) * 100
                        
                        if success_rate < self.thresholds['analyzer_success_rate']['critical']:
                            alerts.append(QualityAlert(
                                'critical', analyzer_name, 'success_rate',
                                success_rate, self.thresholds['analyzer_success_rate']['critical'],
                                f"{analyzer_name} success rate critical: {success_rate:.1f}%"
                            ))
                        elif success_rate < self.thresholds['analyzer_success_rate']['warning']:
                            alerts.append(QualityAlert(
                                'warning', analyzer_name, 'success_rate',
                                success_rate, self.thresholds['analyzer_success_rate']['warning'],
                                f"{analyzer_name} success rate low: {success_rate:.1f}%"
                            ))
                
                conn.close()
                
        except Exception as e:
            logger.error(f"Error checking success rates: {e}")
        
        return alerts
    
    def _get_recent_results(self, hours: int = 1) -> List[Path]:
        """Get recent analysis result files"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_files = []
        
        for json_file in self.results_dir.glob("*.json"):
            if json_file.stat().st_mtime > cutoff_time.timestamp():
                recent_files.append(json_file)
        
        return recent_files
    
    def _get_expected_fields(self, analyzer_name: str) -> List[str]:
        """Get expected fields for each analyzer"""
        expected_fields = {
            'qwen2_vl_temporal': ['description', 'timestamp', 'confidence'],
            'object_detection': ['objects', 'timestamp'],
            'eye_tracking': ['gaze_direction', 'confidence'],
            'speech_rate': ['pitch_hz', 'wpm', 'text'],
            'text_overlay': ['text', 'position', 'confidence']
        }
        
        return expected_fields.get(analyzer_name, [])
    
    async def send_alerts(self, alerts: List[QualityAlert]):
        """Send alerts via configured channels"""
        if not alerts:
            return
        
        # Log all alerts
        for alert in alerts:
            log_message = f"[{alert.level.upper()}] {alert.analyzer}: {alert.message}"
            if alert.level == 'critical':
                logger.critical(log_message)
            else:
                logger.warning(log_message)
        
        # Send email if configured
        if self.email_config['enabled']:
            await self._send_email_alerts(alerts)
        
        # Write to alert file
        alert_file = self.results_dir.parent / 'quality_alerts.json'
        alert_data = []
        
        for alert in alerts:
            alert_data.append({
                'timestamp': alert.timestamp.isoformat(),
                'level': alert.level,
                'analyzer': alert.analyzer,
                'metric': alert.metric,
                'current_value': alert.current_value,
                'threshold': alert.threshold,
                'message': alert.message
            })
        
        with open(alert_file, 'w') as f:
            json.dump(alert_data, f, indent=2)
    
    async def _send_email_alerts(self, alerts: List[QualityAlert]):
        """Send email alerts"""
        try:
            # Group alerts by level
            critical_alerts = [a for a in alerts if a.level == 'critical']
            warning_alerts = [a for a in alerts if a.level == 'warning']
            
            # Create email content
            subject = f"TikTok Analyzer Quality Alert: {len(critical_alerts)} critical, {len(warning_alerts)} warnings"
            
            body = "TikTok Video Analyzer Quality Report\n\n"
            
            if critical_alerts:
                body += "CRITICAL ALERTS:\n"
                for alert in critical_alerts:
                    body += f"- {alert.message}\n"
                body += "\n"
            
            if warning_alerts:
                body += "WARNING ALERTS:\n"
                for alert in warning_alerts:
                    body += f"- {alert.message}\n"
            
            # Send email
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = self.email_config['from_email']
            msg['To'] = ', '.join(self.email_config['to_emails'])
            
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['from_email'], self.email_config['password'])
                server.send_message(msg)
            
            logger.info(f"Sent email alert with {len(alerts)} alerts")
            
        except Exception as e:
            logger.error(f"Failed to send email alerts: {e}")
    
    async def generate_daily_report(self):
        """Generate daily quality report"""
        report = {
            'date': datetime.now().date().isoformat(),
            'summary': {},
            'analyzer_stats': {},
            'recommendations': []
        }
        
        try:
            # Collect 24-hour statistics
            if Path(self.monitoring_db).exists():
                conn = sqlite3.connect(self.monitoring_db)
                cursor = conn.cursor()
                
                # Overall stats
                cursor.execute('''
                    SELECT 
                        COUNT(DISTINCT analyzer_name) as analyzer_count,
                        AVG(avg_processing_time) as avg_processing_time,
                        AVG(duplicate_rate) as avg_duplicate_rate
                    FROM analyzer_stats
                    WHERE timestamp > datetime('now', '-1 day')
                ''')
                
                row = cursor.fetchone()
                report['summary'] = {
                    'active_analyzers': row[0] or 0,
                    'avg_processing_time': row[1] or 0,
                    'avg_duplicate_rate': row[2] or 0
                }
                
                # Per-analyzer stats
                cursor.execute('''
                    SELECT 
                        analyzer_name,
                        COUNT(*) as run_count,
                        AVG(avg_processing_time) as avg_time,
                        AVG(duplicate_rate) as duplicate_rate,
                        SUM(error_count) as total_errors
                    FROM analyzer_stats
                    WHERE timestamp > datetime('now', '-1 day')
                    GROUP BY analyzer_name
                ''')
                
                for row in cursor.fetchall():
                    report['analyzer_stats'][row[0]] = {
                        'runs': row[1],
                        'avg_processing_time': row[2] or 0,
                        'duplicate_rate': row[3] or 0,
                        'errors': row[4] or 0
                    }
                
                conn.close()
            
            # Generate recommendations
            if report['summary']['avg_duplicate_rate'] > 5:
                report['recommendations'].append(
                    "High duplicate rate detected. Consider updating Qwen2-VL prompts."
                )
            
            if report['summary']['avg_processing_time'] > 90:
                report['recommendations'].append(
                    "Processing times exceed target. Consider optimizing batch sizes."
                )
            
            # Save report
            report_file = self.results_dir.parent / f"quality_report_{report['date']}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Generated daily report: {report_file}")
            
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
    
    async def monitor_loop(self):
        """Main monitoring loop"""
        logger.info("Starting quality monitor...")
        
        while True:
            try:
                # Run quality checks
                all_alerts = []
                
                # Check analyzer quality
                analyzer_alerts = await self.check_analyzer_quality()
                all_alerts.extend(analyzer_alerts)
                
                # Check system performance
                system_alerts = await self.check_system_performance()
                all_alerts.extend(system_alerts)
                
                # Check success rates
                success_alerts = await self.check_analyzer_success_rates()
                all_alerts.extend(success_alerts)
                
                # Send alerts if any
                if all_alerts:
                    await self.send_alerts(all_alerts)
                
                # Generate daily report at midnight
                current_hour = datetime.now().hour
                if current_hour == 0:
                    await self.generate_daily_report()
                
                # Wait before next check
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(60)

if __name__ == "__main__":
    monitor = QualityMonitor()
    asyncio.run(monitor.monitor_loop())