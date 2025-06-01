#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT Stage 2: CIFAR-10 リアルタイム進捗監視
Weights & Biases統合版
"""

import os
import time
import json
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# グラフ文字化け防止設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)

class NKATProgressMonitor:
    """NKAT進捗監視システム"""
    
    def __init__(self, use_wandb=False):
        self.use_wandb = use_wandb
        self.log_data = []
        
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                wandb.init(
                    project="NKAT-Stage2", 
                    name=f"cifar10-smoke-{datetime.now().strftime('%H%M')}"
                )
                print("✅ W&B integration enabled")
            except ImportError:
                print("⚠️  wandb not available, using local logging")
                self.use_wandb = False
    
    def check_optuna_progress(self):
        """Optuna trial進捗チェック"""
        results = {}
        
        # JSONファイルから結果読み取り
        json_files = [f for f in os.listdir('.') if f.endswith('config.json')]
        
        if json_files:
            with open(json_files[-1], 'r') as f:
                data = json.load(f)
                results['best_trial'] = data.get('trial_number', 0)
                results['best_accuracy'] = data.get('objective_value', 0)
                results['estimated_full'] = data.get('estimated_full_accuracy', 0)
        
        return results
    
    def monitor_gpu_efficiency(self):
        """GPU効率監視"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            efficiency = (memory_allocated / memory_total) * 100
            
            return {
                'gpu_memory_gb': memory_allocated,
                'gpu_total_gb': memory_total,
                'gpu_efficiency': efficiency,
                'gpu_name': torch.cuda.get_device_name(0)
            }
        return None
    
    def extract_training_metrics(self, log_text):
        """ログテキストから学習メトリクス抽出"""
        import re
        
        # Epoch進捗パターン
        epoch_pattern = r'Epoch (\d+)/(\d+).*?Acc=(\d+\.\d+)%.*?LR=(\d+\.\d+)'
        val_pattern = r'Epoch \d+: Train=(\d+\.\d+)%, Test=(\d+\.\d+)%, Best=(\d+\.\d+)%'
        
        epochs = []
        train_acc = []
        val_acc = []
        learning_rates = []
        
        for match in re.finditer(epoch_pattern, log_text):
            epoch, total, acc, lr = match.groups()
            epochs.append(int(epoch))
            train_acc.append(float(acc))
            learning_rates.append(float(lr))
        
        for match in re.finditer(val_pattern, log_text):
            train_a, val_a, best_a = match.groups()
            val_acc.append(float(val_a))
        
        return {
            'epochs': epochs,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'learning_rates': learning_rates
        }
    
    def create_progress_plot(self, metrics):
        """進捗グラフ作成"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Accuracy Plot
        if metrics['epochs'] and metrics['train_accuracy']:
            ax1.plot(metrics['epochs'], metrics['train_accuracy'], 'b-o', label='Train Accuracy', linewidth=2)
            if metrics['val_accuracy']:
                ax1.plot(metrics['epochs'][:len(metrics['val_accuracy'])], metrics['val_accuracy'], 'r-o', label='Val Accuracy', linewidth=2)
            
            # 目標ライン
            ax1.axhline(y=45, color='green', linestyle='--', alpha=0.7, label='Target: 45%')
            ax1.axhline(y=55, color='orange', linestyle='--', alpha=0.7, label='Optuna Goal: 55%')
            
            ax1.set_title('NKAT CIFAR-10 Training Progress', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy (%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Learning Rate Plot
        if metrics['epochs'] and metrics['learning_rates']:
            ax2.plot(metrics['epochs'], metrics['learning_rates'], 'g-', linewidth=2)
            ax2.set_title('Learning Rate Schedule', fontsize=14)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'nkat_progress_{timestamp}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"📊 Progress plot saved: {filename}")
        
        if self.use_wandb:
            self.wandb.log({"progress_plot": self.wandb.Image(filename)})
        
        return filename
    
    def check_milestone_achievement(self, val_acc, epoch):
        """マイルストーン達成チェック"""
        milestones = {
            3: 22,   # Epoch 3で22%期待
            5: 30,   # Epoch 5で30%期待  
            8: 40,   # Epoch 8で40%期待
            10: 45   # Epoch 10で45%目標
        }
        
        if epoch in milestones:
            target = milestones[epoch]
            achieved = val_acc >= target
            
            status = "✅ ACHIEVED" if achieved else "⚠️ BELOW TARGET"
            print(f"🎯 Epoch {epoch} Milestone: {val_acc:.1f}% (Target: {target}%) - {status}")
            
            if self.use_wandb:
                self.wandb.log({
                    f"milestone_epoch_{epoch}": val_acc,
                    f"milestone_target_{epoch}": target,
                    f"milestone_achieved_{epoch}": achieved
                })
            
            return achieved
        return None
    
    def generate_report(self):
        """総合レポート生成"""
        gpu_info = self.monitor_gpu_efficiency()
        optuna_info = self.check_optuna_progress()
        
        report = f"""
🔍 NKAT Stage 2 Progress Report - {datetime.now().strftime('%H:%M:%S')}
{'='*60}

🎮 GPU Status:
   Device: {gpu_info['gpu_name'] if gpu_info else 'N/A'}
   Memory: {gpu_info['gpu_memory_gb']:.1f}GB / {gpu_info['gpu_total_gb']:.1f}GB
   Efficiency: {gpu_info['gpu_efficiency']:.1f}%

🚀 Optuna Progress:
   Best Trial: #{optuna_info.get('best_trial', 'N/A')}
   Best Accuracy: {optuna_info.get('best_accuracy', 0):.2f}%
   Est. Full Training: {optuna_info.get('estimated_full', 0):.1f}%

📊 Next Milestones:
   - Epoch 3: 22% (Warmup Complete)
   - Epoch 5: 30% (RandAugment Effect)
   - Epoch 8: 40% (CosineLR Descent)
   - Epoch 10: 45% (SMOKE TEST PASS)

🎯 Stage 2 Ultimate Goals:
   ✓ CIFAR-10 Val ≥ 60%
   ✓ λ_theory ≈ 12k (log10 < 0.05)
   ✓ Global TPE ≥ 0.70

{'='*60}
        """
        
        print(report)
        return report

if __name__ == "__main__":
    import sys
    
    # 引数でW&B有効化
    use_wandb = '--wandb' in sys.argv
    
    monitor = NKATProgressMonitor(use_wandb=use_wandb)
    
    if '--continuous' in sys.argv:
        print("🔄 Continuous monitoring mode")
        try:
            while True:
                monitor.generate_report()
                time.sleep(60)  # 1分間隔
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped")
    else:
        monitor.generate_report() 