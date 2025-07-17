#!/usr/bin/env python3
"""
Bulk processor for adding TikTok videos to processing queue
Supports file input, direct URLs, and monitoring
"""

import click
import json
from pathlib import Path
from typing import List
import webbrowser
import time
from datetime import datetime
import requests

from queue_manager import QueueManager
from celery_config import app as celery_app

@click.group()
def cli():
    """TikTok Mass Processing CLI"""
    pass

@cli.command()
@click.option('--file', '-f', type=click.Path(exists=True), help='File containing TikTok URLs (one per line)')
@click.option('--urls', '-u', multiple=True, help='TikTok URLs to process')
@click.option('--priority', '-p', type=int, default=5, help='Priority (1-10, higher = more priority)')
@click.option('--monitor', '-m', is_flag=True, help='Open monitoring dashboard after adding')
@click.option('--dry-run', is_flag=True, help='Show what would be added without actually adding')
def add(file, urls, priority, monitor, dry_run):
    """Add TikTok videos to processing queue"""
    
    queue_manager = QueueManager()
    
    # Collect URLs
    all_urls = list(urls)
    
    if file:
        click.echo(f"📄 Reading URLs from {file}...")
        with open(file, 'r') as f:
            file_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            all_urls.extend(file_urls)
            click.echo(f"   Found {len(file_urls)} URLs in file")
    
    if not all_urls:
        click.echo("❌ No URLs provided! Use -u URL or -f filename", err=True)
        return
    
    # Validate URLs
    valid_urls = []
    for url in all_urls:
        if 'tiktok.com' in url:
            valid_urls.append(url)
        else:
            click.echo(f"⚠️  Invalid URL (not TikTok): {url}", err=True)
    
    if not valid_urls:
        click.echo("❌ No valid TikTok URLs found!", err=True)
        return
    
    click.echo(f"\n📊 Summary:")
    click.echo(f"   Total URLs: {len(valid_urls)}")
    click.echo(f"   Priority: {priority}")
    
    if dry_run:
        click.echo("\n🔍 Dry run - URLs that would be added:")
        for url in valid_urls[:10]:  # Show first 10
            click.echo(f"   - {url}")
        if len(valid_urls) > 10:
            click.echo(f"   ... and {len(valid_urls) - 10} more")
        return
    
    # Add to queue
    click.echo(f"\n🚀 Adding {len(valid_urls)} URLs to queue...")
    
    with click.progressbar(valid_urls, label='Adding URLs') as urls_bar:
        results = {}
        for url in urls_bar:
            success = queue_manager.add_url(url, priority)
            results[url] = success
    
    # Summary
    added = sum(1 for v in results.values() if v)
    skipped = len(results) - added
    
    click.echo(f"\n✅ Successfully added: {added}")
    if skipped > 0:
        click.echo(f"⏭️  Skipped (already processed/processing): {skipped}")
    
    # Show current queue stats
    stats = queue_manager.get_queue_stats()
    click.echo("\n📈 Current Queue Status:")
    click.echo(f"   Download queue: {stats.get('queue_download', 0)}")
    click.echo(f"   Processing queue: {stats.get('queue_processing', 0)}")
    click.echo(f"   Priority queue: {stats.get('queue_priority', 0)}")
    click.echo(f"   Failed queue: {stats.get('queue_failed', 0)}")
    click.echo(f"   Total processed: {stats.get('processed', 0)}")
    
    if monitor:
        click.echo("\n🖥️  Opening monitoring dashboard...")
        webbrowser.open('http://localhost:5000')

@cli.command()
def status():
    """Show current queue and worker status"""
    
    queue_manager = QueueManager()
    
    # Get queue stats
    stats = queue_manager.get_queue_stats()
    
    click.echo("📊 Queue Status")
    click.echo("=" * 50)
    for key, value in sorted(stats.items()):
        click.echo(f"{key:.<30} {value:>10,}")
    
    # Get worker status via Celery
    click.echo("\n👷 Worker Status")
    click.echo("=" * 50)
    
    try:
        # Get active tasks
        active = celery_app.control.inspect().active()
        if active:
            total_active = sum(len(tasks) for tasks in active.values())
            click.echo(f"Active tasks: {total_active}")
            
            for worker, tasks in active.items():
                if tasks:
                    click.echo(f"\n{worker}:")
                    for task in tasks[:3]:  # Show first 3 tasks
                        click.echo(f"  - {task['name']} (started: {task.get('time_start', 'unknown')})")
        else:
            click.echo("No active workers found")
            
        # Get registered tasks
        registered = celery_app.control.inspect().registered()
        if registered:
            click.echo(f"\nRegistered workers: {len(registered)}")
            
    except Exception as e:
        click.echo(f"⚠️  Could not connect to Celery: {e}", err=True)
    
    # Get recent completions
    click.echo("\n🎬 Recent Completions")
    click.echo("=" * 50)
    
    try:
        response = requests.get('http://localhost:5000/api/metrics')
        if response.ok:
            data = response.json()
            recent = data.get('recent_completions', [])
            
            if recent:
                for item in recent[:5]:
                    time_str = datetime.fromisoformat(item['analyzed_at']).strftime('%H:%M:%S')
                    click.echo(f"{time_str} - {item['tiktok_id']} - {item['successful_analyzers']} analyzers - {item['total_processing_time']:.1f}s")
            else:
                click.echo("No recent completions")
        else:
            click.echo("⚠️  Could not fetch recent completions", err=True)
    except:
        pass

@cli.command()
@click.option('--max-retries', type=int, default=3, help='Maximum retry attempts')
def requeue(max_retries):
    """Requeue failed tasks"""
    
    queue_manager = QueueManager()
    
    # Get current failed count
    stats = queue_manager.get_queue_stats()
    failed_count = stats.get('queue_failed', 0)
    
    if failed_count == 0:
        click.echo("✅ No failed tasks to requeue")
        return
    
    click.echo(f"🔄 Found {failed_count} failed tasks")
    
    # Requeue
    requeued = queue_manager.requeue_failed(max_retries=max_retries)
    
    click.echo(f"✅ Requeued {requeued} tasks (max retries: {max_retries})")
    
    if requeued < failed_count:
        click.echo(f"⚠️  {failed_count - requeued} tasks exceeded retry limit")

@cli.command()
@click.option('--days', type=int, default=7, help='Remove tasks older than N days')
def cleanup(days):
    """Clean up old tasks from queues"""
    
    queue_manager = QueueManager()
    
    click.echo(f"🧹 Cleaning up tasks older than {days} days...")
    
    removed = queue_manager.cleanup_old_tasks(days=days)
    
    click.echo(f"✅ Removed {removed} old tasks")

@cli.command()
@click.option('--format', type=click.Choice(['json', 'csv']), default='json', help='Export format')
@click.option('--output', '-o', type=click.Path(), help='Output file')
def export(format, output):
    """Export queue statistics"""
    
    queue_manager = QueueManager()
    
    if not output:
        output = f"queue_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
    
    click.echo(f"📤 Exporting to {output}...")
    
    if format == 'json':
        queue_manager.export_stats_json(output)
    else:
        # Export CSV via API
        try:
            response = requests.get('http://localhost:5000/api/export/csv')
            if response.ok:
                with open(output, 'w') as f:
                    f.write(response.text)
            else:
                click.echo("❌ Export failed", err=True)
                return
        except Exception as e:
            click.echo(f"❌ Export error: {e}", err=True)
            return
    
    click.echo(f"✅ Exported to {output}")

@cli.command()
def monitor():
    """Open monitoring dashboard"""
    
    click.echo("🖥️  Opening monitoring dashboard...")
    webbrowser.open('http://localhost:5000')
    
    click.echo("\nMonitoring URLs:")
    click.echo("  Dashboard: http://localhost:5000")
    click.echo("  Flower: http://localhost:5555")

@cli.command()
@click.confirmation_option(prompt='Are you sure you want to clear all queues?')
def clear():
    """Clear all queues (dangerous!)"""
    
    queue_manager = QueueManager()
    
    # Get current stats before clearing
    stats = queue_manager.get_queue_stats()
    total = sum(v for k, v in stats.items() if k.startswith('queue_'))
    
    click.echo(f"🗑️  Clearing {total} items from all queues...")
    
    queue_manager.clear_all_queues()
    
    click.echo("✅ All queues cleared!")

@cli.command()
@click.argument('url')
@click.option('--priority', '-p', type=int, default=10, help='Priority level')
def process_now(url, priority):
    """Process a single video immediately with high priority"""
    
    if 'tiktok.com' not in url:
        click.echo("❌ Invalid TikTok URL", err=True)
        return
    
    queue_manager = QueueManager()
    
    click.echo(f"🚀 Adding {url} with HIGH priority ({priority})...")
    
    success = queue_manager.add_url(url, priority)
    
    if success:
        click.echo("✅ Added to priority queue!")
        click.echo("📊 Monitoring progress at http://localhost:5000")
        
        # Optionally open monitoring
        if click.confirm("Open monitoring dashboard?"):
            webbrowser.open('http://localhost:5000')
    else:
        click.echo("⚠️  URL already in queue or processed")

if __name__ == '__main__':
    cli()