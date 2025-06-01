#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT Stage 2: リアルタイム進捗モニター
バックグラウンド実行の状況をチェック
"""

import os
import time
import psutil
import torch
import json
from datetime import datetime

def check_gpu_usage():
    """GPU使用状況チェック"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated(0) / 1024**3  # GB
        gpu_cached = torch.cuda.memory_reserved(0) / 1024**3   # GB
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return {
            "allocated": gpu_memory,
            "cached": gpu_cached, 
            "total": gpu_total,
            "utilization": (gpu_memory / gpu_total) * 100
        }
    return None

def check_training_progress():
    """学習進捗チェック"""
    
    # Check for saved models (progress indicator)
    model_files = []
    if os.path.exists("nkat_models"):
        for f in os.listdir("nkat_models"):
            if f.endswith(".pth"):
                model_files.append(f)
    
    # Check for log files
    log_files = []
    for f in os.listdir("."):
        if "log" in f.lower() and f.endswith(".txt"):
            log_files.append(f)
    
    return {
        "models_saved": len(model_files),
        "model_files": model_files,
        "log_files": log_files,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }

def check_processes():
    """Pythonプロセス状況チェック"""
    python_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if 'nkat' in cmdline.lower() or 'smoke' in cmdline.lower():
                    python_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline,
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_percent': proc.info['memory_percent']
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    return python_processes

def monitor_status():
    """総合ステータスモニタリング"""
    
    print("🔍 NKAT Stage 2 Progress Monitor")
    print("=" * 50)
    
    # GPU Status
    gpu_info = check_gpu_usage()
    if gpu_info:
        print(f"🎮 GPU Status:")
        print(f"   Memory: {gpu_info['allocated']:.1f}GB / {gpu_info['total']:.1f}GB")
        print(f"   Utilization: {gpu_info['utilization']:.1f}%")
        
        if gpu_info['utilization'] > 50:
            print("   ✅ Training likely in progress")
        else:
            print("   ⏸️  Low GPU usage - training may be idle")
    else:
        print("🎮 GPU: Not available or not in use")
    
    print()
    
    # Training Progress
    progress = check_training_progress()
    print(f"📊 Training Progress:")
    print(f"   Models saved: {progress['models_saved']}")
    print(f"   Log files: {len(progress['log_files'])}")
    print(f"   Last check: {progress['timestamp']}")
    
    if progress['models_saved'] > 0:
        print(f"   Latest model: {progress['model_files'][-1]}")
    
    print()
    
    # Process Status
    processes = check_processes()
    print(f"🔧 Python Processes:")
    if processes:
        for proc in processes:
            print(f"   PID {proc['pid']}: CPU {proc['cpu_percent']:.1f}%, RAM {proc['memory_percent']:.1f}%")
            print(f"   CMD: {proc['cmdline']}")
    else:
        print("   No NKAT-related processes found")
    
    print("=" * 50)

def continuous_monitor(interval=30):
    """連続モニタリング"""
    print("🔄 Continuous monitoring started (Ctrl+C to stop)")
    print(f"⏱️  Update interval: {interval} seconds")
    
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
            monitor_status()
            print(f"\n⏰ Next update in {interval} seconds...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        continuous_monitor(interval)
    else:
        monitor_status()
        print("\n💡 Tip: Use 'py -3 monitor_progress.py --continuous 30' for live monitoring") 