#!/usr/bin/env python3
"""
Real-time GPU Monitoring Tool for TikTok Production System
Shows GPU utilization, memory usage, and analyzer performance
"""
import torch
import time
import psutil
import curses
import subprocess
import json
from datetime import datetime
import numpy as np
from collections import deque
import threading
import sys

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not available, using fallback GPU monitoring")


class GPUMonitor:
    def __init__(self):
        self.history_size = 60  # 60 seconds of history
        self.gpu_util_history = deque(maxlen=self.history_size)
        self.gpu_mem_history = deque(maxlen=self.history_size)
        self.fps_history = deque(maxlen=self.history_size)
        self.analyzer_stats = {}
        self.start_time = time.time()
        self.last_log_check = 0
        
        # Initialize NVML if available
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.nvml_initialized = True
            except:
                self.nvml_initialized = False
        else:
            self.nvml_initialized = False
            
    def get_gpu_stats(self):
        """Get current GPU statistics"""
        stats = {
            'util': 0,
            'memory_used': 0,
            'memory_total': 0,
            'memory_percent': 0,
            'temperature': 0,
            'power': 0,
            'power_limit': 0
        }
        
        if self.nvml_initialized:
            try:
                # GPU Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                stats['util'] = util.gpu
                
                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                stats['memory_used'] = mem_info.used / (1024**3)  # GB
                stats['memory_total'] = mem_info.total / (1024**3)  # GB
                stats['memory_percent'] = (mem_info.used / mem_info.total) * 100
                
                # Temperature
                stats['temperature'] = pynvml.nvmlDeviceGetTemperature(
                    self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU
                )
                
                # Power
                stats['power'] = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000  # W
                stats['power_limit'] = pynvml.nvmlDeviceGetPowerManagementLimit(self.gpu_handle) / 1000  # W
                
            except Exception as e:
                pass
                
        elif torch.cuda.is_available():
            # Fallback to PyTorch
            stats['memory_used'] = torch.cuda.memory_allocated() / (1024**3)
            stats['memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            stats['memory_percent'] = (stats['memory_used'] / stats['memory_total']) * 100
            
            # Try nvidia-smi for utilization
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    stats['util'] = int(result.stdout.strip())
            except:
                pass
                
        return stats
        
    def get_analyzer_stats(self):
        """Parse recent logs for analyzer performance"""
        log_file = "/home/user/tiktok_production/logs/stable_api.log"
        
        try:
            # Only check new log entries
            current_time = time.time()
            if current_time - self.last_log_check < 1:  # Check every second
                return
                
            self.last_log_check = current_time
            
            # Read last 100 lines of log
            with subprocess.Popen(
                ['tail', '-n', '100', log_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            ) as proc:
                lines = proc.stdout.readlines()
                
            for line in lines:
                # Parse analyzer completion times
                if "completed in" in line and "seconds" in line:
                    try:
                        # Extract analyzer name and time
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.endswith(':') and i+3 < len(parts) and parts[i+2] == 'in':
                                analyzer = part[:-1]
                                time_taken = float(parts[i+3])
                                self.analyzer_stats[analyzer] = {
                                    'last_time': time_taken,
                                    'timestamp': current_time
                                }
                    except:
                        pass
                        
        except Exception as e:
            pass
            
    def draw_screen(self, stdscr):
        """Draw the monitoring interface"""
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)   # Non-blocking input
        
        # Colors
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
        
        while True:
            try:
                # Get current stats
                gpu_stats = self.get_gpu_stats()
                self.get_analyzer_stats()
                
                # Update history
                self.gpu_util_history.append(gpu_stats['util'])
                self.gpu_mem_history.append(gpu_stats['memory_percent'])
                
                # Clear screen
                stdscr.clear()
                height, width = stdscr.getmaxyx()
                
                # Header
                header = "🚀 TikTok Production GPU Monitor - Target: 90% Utilization"
                stdscr.addstr(0, (width - len(header)) // 2, header, curses.A_BOLD | curses.color_pair(4))
                
                # GPU Stats
                y = 2
                stdscr.addstr(y, 2, "GPU Statistics:", curses.A_BOLD)
                y += 1
                stdscr.addstr(y, 4, f"Utilization: {gpu_stats['util']:3d}%", 
                            curses.color_pair(self._get_util_color(gpu_stats['util'])))
                
                # Draw utilization bar
                bar_width = min(50, width - 25)
                filled = int(bar_width * gpu_stats['util'] / 100)
                bar = "█" * filled + "░" * (bar_width - filled)
                stdscr.addstr(y, 25, f"[{bar}]")
                
                y += 2
                stdscr.addstr(y, 4, f"Memory: {gpu_stats['memory_used']:.1f}/{gpu_stats['memory_total']:.1f} GB "
                            f"({gpu_stats['memory_percent']:.1f}%)")
                
                # Memory bar
                filled = int(bar_width * gpu_stats['memory_percent'] / 100)
                bar = "█" * filled + "░" * (bar_width - filled)
                stdscr.addstr(y, 25, f"[{bar}]")
                
                y += 2
                if gpu_stats['temperature'] > 0:
                    stdscr.addstr(y, 4, f"Temperature: {gpu_stats['temperature']}°C")
                    y += 1
                    
                if gpu_stats['power'] > 0:
                    stdscr.addstr(y, 4, f"Power: {gpu_stats['power']:.0f}W / {gpu_stats['power_limit']:.0f}W")
                    y += 1
                    
                # Utilization History Graph
                y += 2
                stdscr.addstr(y, 2, "GPU Utilization History (60s):", curses.A_BOLD)
                y += 1
                
                if len(self.gpu_util_history) > 1:
                    graph_height = 10
                    graph_width = min(60, width - 10)
                    
                    # Draw graph
                    for h in range(graph_height):
                        line = ""
                        threshold = 100 - (h * 10)
                        
                        for i in range(min(len(self.gpu_util_history), graph_width)):
                            idx = -graph_width + i if len(self.gpu_util_history) > graph_width else i
                            val = self.gpu_util_history[idx]
                            
                            if val >= threshold:
                                line += "█"
                            else:
                                line += " "
                                
                        # Add scale
                        stdscr.addstr(y + h, 2, f"{threshold:3d}% |{line}|")
                        
                    y += graph_height + 1
                    
                    # Average and target
                    avg_util = np.mean(list(self.gpu_util_history))
                    stdscr.addstr(y, 2, f"Average: {avg_util:.1f}%", 
                                curses.color_pair(self._get_util_color(avg_util)))
                    stdscr.addstr(y, 20, f"Target: 90%", curses.color_pair(1))
                    
                # Active Analyzers
                y += 3
                if y < height - 10:
                    stdscr.addstr(y, 2, "Recent Analyzer Performance:", curses.A_BOLD)
                    y += 1
                    
                    # Sort by most recent
                    recent_analyzers = sorted(
                        self.analyzer_stats.items(),
                        key=lambda x: x[1]['timestamp'],
                        reverse=True
                    )[:10]
                    
                    for analyzer, stats in recent_analyzers:
                        if y < height - 2:
                            age = time.time() - stats['timestamp']
                            if age < 60:  # Only show from last minute
                                stdscr.addstr(y, 4, f"{analyzer[:30]:30} {stats['last_time']:6.1f}s")
                                y += 1
                                
                # System stats
                cpu_percent = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory()
                
                stdscr.addstr(height-2, 2, 
                            f"CPU: {cpu_percent:4.1f}% | RAM: {mem.percent:4.1f}% | "
                            f"Uptime: {int(time.time() - self.start_time)}s")
                
                # Instructions
                stdscr.addstr(height-1, 2, "Press 'q' to quit, 'r' to reset history", curses.color_pair(5))
                
                # Refresh
                stdscr.refresh()
                
                # Check for input
                key = stdscr.getch()
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.gpu_util_history.clear()
                    self.gpu_mem_history.clear()
                    self.analyzer_stats.clear()
                    
                time.sleep(1)  # Update every second
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                # Handle resize and other errors gracefully
                pass
                
    def _get_util_color(self, util):
        """Get color based on utilization percentage"""
        if util >= 85:
            return 1  # Green - Good!
        elif util >= 60:
            return 2  # Yellow - OK
        else:
            return 3  # Red - Too low
            
    def cleanup(self):
        """Cleanup resources"""
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


def run_simple_monitor():
    """Simple monitor without curses (for non-terminal environments)"""
    monitor = GPUMonitor()
    
    print("GPU Monitor - Simple Mode (Ctrl+C to exit)")
    print("=" * 60)
    
    try:
        while True:
            stats = monitor.get_gpu_stats()
            
            # Clear screen (works on most terminals)
            print("\033[2J\033[H", end="")
            
            print(f"GPU Utilization: {stats['util']:3d}% {'█' * (stats['util']//2)}")
            print(f"GPU Memory: {stats['memory_used']:.1f}/{stats['memory_total']:.1f} GB ({stats['memory_percent']:.1f}%)")
            
            if stats['temperature'] > 0:
                print(f"Temperature: {stats['temperature']}°C")
            if stats['power'] > 0:
                print(f"Power: {stats['power']:.0f}W / {stats['power_limit']:.0f}W")
                
            print(f"\nTarget Utilization: 90%")
            
            if stats['util'] < 85:
                print("⚠️  GPU utilization below target!")
                
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
    finally:
        monitor.cleanup()


def main():
    """Main entry point"""
    # Check if we're in a proper terminal
    if sys.stdout.isatty() and sys.stdin.isatty():
        # Use curses interface
        monitor = GPUMonitor()
        try:
            curses.wrapper(monitor.draw_screen)
        finally:
            monitor.cleanup()
    else:
        # Fallback to simple mode
        run_simple_monitor()


# Export functions for API compatibility
def gpu_monitor():
    """Simple GPU monitor function for API"""
    monitor = GPUMonitor()
    return monitor.get_gpu_stats()

def log_gpu_memory(message=""):
    """Log GPU memory usage with optional message"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[GPU Memory] {message}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    else:
        print(f"[GPU Memory] {message}: No GPU available")

if __name__ == "__main__":
    main()