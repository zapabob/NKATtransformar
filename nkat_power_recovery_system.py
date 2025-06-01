#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT Power Recovery System
RTX3080 CUDA対応 電源断リカバリーシステム
"""

import os
import json
import torch
import pickle
import shutil
import psutil
import hashlib
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm
import time

class NKATPowerRecoverySystem:
    """電源断リカバリーシステム"""
    
    def __init__(self, 
                 checkpoint_dir: str = "checkpoints",
                 backup_dir: str = "backups", 
                 auto_save_interval: int = 300):  # 5分間隔
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.backup_dir = Path(backup_dir)
        self.auto_save_interval = auto_save_interval
        
        # ディレクトリ作成
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
        
        self.recovery_metadata = {
            'session_id': self.generate_session_id(),
            'start_time': datetime.now().isoformat(),
            'cuda_device': None,
            'last_checkpoint': None,
            'recovery_count': 0
        }
        
        self.initialize_cuda()
        self.setup_recovery_hooks()
        
        print(f"🔋 Power Recovery System initialized")
        print(f"   Session ID: {self.recovery_metadata['session_id']}")
        print(f"   CUDA Device: {self.recovery_metadata['cuda_device']}")
        print(f"   Checkpoint Dir: {self.checkpoint_dir}")
        
    def initialize_cuda(self):
        """CUDA環境初期化"""
        if torch.cuda.is_available():
            device_id = 0  # RTX3080
            torch.cuda.set_device(device_id)
            device_name = torch.cuda.get_device_name(device_id)
            self.recovery_metadata['cuda_device'] = device_name
            
            # メモリキャッシュクリア
            torch.cuda.empty_cache()
            
            print(f"✅ CUDA initialized: {device_name}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print("⚠️ CUDA not available")
            
    def generate_session_id(self) -> str:
        """セッションID生成"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"nkat_{timestamp}_{unique_id}"
    
    def setup_recovery_hooks(self):
        """リカバリーフック設定"""
        import signal
        import atexit
        
        # シグナルハンドラー設定
        signal.signal(signal.SIGINT, self.emergency_save_handler)
        signal.signal(signal.SIGTERM, self.emergency_save_handler)
        
        # 終了時処理
        atexit.register(self.cleanup_on_exit)
        
    def emergency_save_handler(self, signum, frame):
        """緊急保存ハンドラー"""
        print(f"\n🚨 Emergency save triggered (signal {signum})")
        self.force_checkpoint("emergency_interrupt")
        exit(0)
        
    def cleanup_on_exit(self):
        """終了時クリーンアップ"""
        print("\n🧹 Cleaning up CUDA memory...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def save_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[Any],
                       epoch: int,
                       best_accuracy: float,
                       loss: float,
                       metrics: Dict[str, Any],
                       checkpoint_type: str = "auto") -> str:
        """チェックポイント保存"""
        
        checkpoint_data = {
            'session_id': self.recovery_metadata['session_id'],
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_accuracy': best_accuracy,
            'current_loss': loss,
            'metrics': metrics,
            'cuda_device': self.recovery_metadata['cuda_device'],
            'checkpoint_type': checkpoint_type,
            'recovery_count': self.recovery_metadata['recovery_count']
        }
        
        # ファイル名生成
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_name = f"nkat_checkpoint_{checkpoint_type}_epoch{epoch:03d}_{timestamp}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # 保存実行
        try:
            # 一時ファイルに保存後、アトミック移動
            temp_path = checkpoint_path.with_suffix('.tmp')
            torch.save(checkpoint_data, temp_path)
            shutil.move(temp_path, checkpoint_path)
            
            # メタデータ更新
            self.recovery_metadata['last_checkpoint'] = str(checkpoint_path)
            self.save_recovery_metadata()
            
            # バックアップ作成（最良性能時）
            if checkpoint_type == "best":
                backup_path = self.backup_dir / f"best_{checkpoint_name}"
                shutil.copy2(checkpoint_path, backup_path)
                
            print(f"💾 Checkpoint saved: {checkpoint_name}")
            print(f"   Epoch: {epoch}, Accuracy: {best_accuracy:.2f}%, Loss: {loss:.4f}")
            
            # 古いチェックポイント削除（最新5個保持）
            self.cleanup_old_checkpoints()
            
            return str(checkpoint_path)
            
        except Exception as e:
            print(f"❌ Checkpoint save failed: {e}")
            traceback.print_exc()
            return None
            
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """チェックポイント読み込み"""
        
        if checkpoint_path is None:
            # 最新チェックポイント自動検索
            checkpoint_path = self.find_latest_checkpoint()
            
        if not checkpoint_path or not Path(checkpoint_path).exists():
            print("📁 No checkpoint found for recovery")
            return None
            
        try:
            print(f"🔄 Loading checkpoint: {Path(checkpoint_path).name}")
            
            # CUDA利用可能性チェック
            map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
            checkpoint_data = torch.load(checkpoint_path, map_location=map_location)
            
            # 互換性確認
            if checkpoint_data.get('cuda_device') != self.recovery_metadata['cuda_device']:
                print(f"⚠️ Device mismatch: {checkpoint_data.get('cuda_device')} -> {self.recovery_metadata['cuda_device']}")
                
            # リカバリーカウント更新
            self.recovery_metadata['recovery_count'] += 1
            
            print(f"✅ Checkpoint loaded successfully")
            print(f"   Session: {checkpoint_data.get('session_id', 'Unknown')}")
            print(f"   Epoch: {checkpoint_data.get('epoch', 0)}")
            print(f"   Best Accuracy: {checkpoint_data.get('best_accuracy', 0):.2f}%")
            print(f"   Recovery Count: {self.recovery_metadata['recovery_count']}")
            
            return checkpoint_data
            
        except Exception as e:
            print(f"❌ Checkpoint load failed: {e}")
            traceback.print_exc()
            return None
            
    def find_latest_checkpoint(self) -> Optional[str]:
        """最新チェックポイント検索"""
        checkpoint_files = list(self.checkpoint_dir.glob("nkat_checkpoint_*.pth"))
        
        if not checkpoint_files:
            return None
            
        # 最新ファイル取得
        latest_file = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        return str(latest_file)
        
    def force_checkpoint(self, reason: str = "manual"):
        """強制チェックポイント（緊急時用）"""
        print(f"🚨 Force checkpoint: {reason}")
        
        # 現在のメモリ状態保存
        emergency_data = {
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'session_id': self.recovery_metadata['session_id'],
            'recovery_count': self.recovery_metadata['recovery_count'],
            'cuda_memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            'system_memory_percent': psutil.virtual_memory().percent
        }
        
        emergency_file = self.checkpoint_dir / f"emergency_{reason}_{datetime.now().strftime('%H%M%S')}.json"
        with open(emergency_file, 'w') as f:
            json.dump(emergency_data, f, indent=2)
            
        print(f"🆘 Emergency state saved: {emergency_file.name}")
        
    def save_recovery_metadata(self):
        """リカバリーメタデータ保存"""
        metadata_file = self.checkpoint_dir / "recovery_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.recovery_metadata, f, indent=2)
            
    def auto_checkpoint_wrapper(self, 
                               save_func,
                               model: torch.nn.Module,
                               optimizer: torch.optim.Optimizer,
                               scheduler: Optional[Any],
                               epoch: int,
                               best_accuracy: float,
                               loss: float,
                               metrics: Dict[str, Any]):
        """自動チェックポイント実行ラッパー"""
        
        # 前回保存からの経過時間チェック
        current_time = time.time()
        last_save_time = getattr(self, '_last_auto_save', 0)
        
        if current_time - last_save_time >= self.auto_save_interval:
            checkpoint_path = save_func(
                model, optimizer, scheduler, epoch, 
                best_accuracy, loss, metrics, "auto"
            )
            self._last_auto_save = current_time
            return checkpoint_path
        return None
        
    def recovery_training_loop(self, 
                              model: torch.nn.Module,
                              optimizer: torch.optim.Optimizer,
                              scheduler: Optional[Any],
                              train_loader,
                              val_loader,
                              epochs: int,
                              device: torch.device):
        """リカバリー対応トレーニングループ"""
        
        print("🚀 Starting recovery-enabled training loop")
        
        # チェックポイントからの復元試行
        checkpoint_data = self.load_checkpoint()
        start_epoch = 0
        best_accuracy = 0
        
        if checkpoint_data:
            model.load_state_dict(checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            if scheduler and checkpoint_data.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            start_epoch = checkpoint_data['epoch'] + 1
            best_accuracy = checkpoint_data['best_accuracy']
            
            print(f"🔄 Resumed from epoch {start_epoch}")
            
        # トレーニング実行
        for epoch in range(start_epoch, epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{epochs} (Recovery Count: {self.recovery_metadata['recovery_count']})")
            print(f"{'='*50}")
            
            try:
                # Training phase
                model.train()
                train_loss = 0
                correct = 0
                total = 0
                
                progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
                
                for batch_idx, (data, target) in enumerate(progress_bar):
                    data, target = data.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = torch.nn.functional.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
                    
                    # プログレスバー更新
                    accuracy = 100. * correct / total
                    progress_bar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{accuracy:.2f}%'
                    })
                    
                    # メモリ使用量監視
                    if torch.cuda.is_available() and batch_idx % 100 == 0:
                        memory_used = torch.cuda.memory_allocated() / 1024**3
                        if memory_used > 10:  # 10GB超過時警告
                            print(f"\n⚠️ High GPU memory usage: {memory_used:.1f}GB")
                            
                # Validation phase
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for data, target in tqdm(val_loader, desc="Validation"):
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        val_loss += torch.nn.functional.cross_entropy(output, target).item()
                        pred = output.argmax(dim=1)
                        val_correct += pred.eq(target).sum().item()
                        val_total += target.size(0)
                        
                train_acc = 100. * correct / total
                val_acc = 100. * val_correct / val_total
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                
                print(f"\nEpoch {epoch+1} Results:")
                print(f"  Train: Loss={avg_train_loss:.4f}, Acc={train_acc:.2f}%")
                print(f"  Val:   Loss={avg_val_loss:.4f}, Acc={val_acc:.2f}%")
                
                # Best model更新
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    checkpoint_type = "best"
                else:
                    checkpoint_type = "regular"
                    
                # チェックポイント保存
                metrics = {
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss
                }
                
                self.save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    best_accuracy, avg_val_loss, metrics, checkpoint_type
                )
                
                # スケジューラーステップ
                if scheduler:
                    scheduler.step()
                    
            except Exception as e:
                print(f"\n❌ Training error at epoch {epoch}: {e}")
                traceback.print_exc()
                
                # 緊急保存
                self.force_checkpoint(f"training_error_epoch_{epoch}")
                
                # CUDA メモリクリア
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                print("🔄 Attempting to continue training...")
                continue
                
        print(f"\n✅ Training completed! Best accuracy: {best_accuracy:.2f}%")
        return best_accuracy
        
    def cleanup_old_checkpoints(self, keep_count: int = 5):
        """古いチェックポイント削除"""
        checkpoint_files = list(self.checkpoint_dir.glob("nkat_checkpoint_auto_*.pth"))
        
        if len(checkpoint_files) > keep_count:
            # 古いファイルから削除
            old_files = sorted(checkpoint_files, key=lambda p: p.stat().st_mtime)[:-keep_count]
            for old_file in old_files:
                old_file.unlink()
                print(f"🗑️ Removed old checkpoint: {old_file.name}")
                
    def generate_recovery_report(self) -> str:
        """リカバリーレポート生成"""
        checkpoint_files = list(self.checkpoint_dir.glob("nkat_checkpoint_*.pth"))
        emergency_files = list(self.checkpoint_dir.glob("emergency_*.json"))
        
        report = f"""
🔋 NKAT Power Recovery System Report
{'='*60}

📊 Session Information:
   Session ID: {self.recovery_metadata['session_id']}
   Start Time: {self.recovery_metadata['start_time']}
   Recovery Count: {self.recovery_metadata['recovery_count']}
   CUDA Device: {self.recovery_metadata['cuda_device']}

💾 Checkpoint Status:
   Total Checkpoints: {len(checkpoint_files)}
   Latest Checkpoint: {Path(self.recovery_metadata.get('last_checkpoint', 'None')).name if self.recovery_metadata.get('last_checkpoint') else 'None'}
   Emergency Saves: {len(emergency_files)}

🔧 System Status:
   CUDA Available: {torch.cuda.is_available()}
   GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.1f}GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB
   CPU Memory: {psutil.virtual_memory().percent:.1f}%

📁 Storage:
   Checkpoint Dir: {self.checkpoint_dir}
   Backup Dir: {self.backup_dir}
   Auto Save Interval: {self.auto_save_interval}s

{'='*60}
        """
        
        print(report)
        return report

if __name__ == "__main__":
    # テスト実行
    recovery_system = NKATPowerRecoverySystem()
    recovery_system.generate_recovery_report() 