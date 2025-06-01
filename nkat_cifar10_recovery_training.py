#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT CIFAR-10 Recovery Training System
RTX3080 CUDA対応 電源断リカバリー統合版
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# リカバリーシステムインポート
from nkat_power_recovery_system import NKATPowerRecoverySystem

# グラフ文字化け防止設定（英語表記）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)

class NKATCifarModel(nn.Module):
    """NKAT CIFAR-10 モデル"""
    
    def __init__(self, num_classes=10):
        super(NKATCifarModel, self).__init__()
        
        # Feature Extraction
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class NKATCifarTrainer:
    """NKAT CIFAR-10 リカバリー対応トレーナー"""
    
    def __init__(self, 
                 batch_size=128,
                 learning_rate=0.001,
                 weight_decay=1e-4,
                 epochs=50,
                 checkpoint_dir="checkpoints_cifar10",
                 data_dir="./data"):
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.data_dir = data_dir
        
        # Device設定 (RTX3080)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🎮 Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # リカバリーシステム初期化
        self.recovery_system = NKATPowerRecoverySystem(
            checkpoint_dir=checkpoint_dir,
            backup_dir=f"{checkpoint_dir}_backup",
            auto_save_interval=300  # 5分間隔
        )
        
        # データセット準備
        self.setup_datasets()
        
        # モデル、オプティマイザー、スケジューラー準備
        self.setup_model()
        
        print("✅ NKAT CIFAR-10 Trainer initialized with power recovery")
        
    def setup_datasets(self):
        """データセット設定"""
        print("📦 Setting up CIFAR-10 datasets...")
        
        # データ変換
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        
        # データセット読み込み
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=train_transform
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=test_transform
        )
        
        # データローダー
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=4, pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Test samples: {len(test_dataset)}")
        print(f"   Batch size: {self.batch_size}")
        
    def setup_model(self):
        """モデル、オプティマイザー、スケジューラー設定"""
        print("🧠 Setting up model and optimizers...")
        
        # モデル
        self.model = NKATCifarModel(num_classes=10).to(self.device)
        
        # オプティマイザー
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # スケジューラー
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )
        
        # パラメータ数表示
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
    def train_with_recovery(self):
        """リカバリー対応トレーニング実行"""
        print("\n🚀 Starting NKAT CIFAR-10 training with power recovery")
        print(f"   Target epochs: {self.epochs}")
        print(f"   Recovery system active: ✅")
        
        try:
            # リカバリーシステムのトレーニングループ使用
            best_accuracy = self.recovery_system.recovery_training_loop(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                train_loader=self.train_loader,
                val_loader=self.test_loader,
                epochs=self.epochs,
                device=self.device
            )
            
            print(f"\n🎉 Training completed successfully!")
            print(f"   Final best accuracy: {best_accuracy:.2f}%")
            
            # 最終レポート生成
            self.recovery_system.generate_recovery_report()
            
            return best_accuracy
            
        except Exception as e:
            print(f"\n❌ Critical training error: {e}")
            
            # 緊急保存
            self.recovery_system.force_checkpoint("critical_error")
            
            print("🔄 You can resume training by running this script again")
            return None
            
    def evaluate_model(self):
        """モデル評価"""
        print("\n📊 Evaluating final model...")
        
        self.model.eval()
        correct = 0
        total = 0
        class_correct = [0] * 10
        class_total = [0] * 10
        
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        with torch.no_grad():
            for data, targets in tqdm(self.test_loader, desc="Evaluating"):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # クラス別精度
                c = (predicted == targets).squeeze()
                for i in range(targets.size(0)):
                    label = targets[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        overall_accuracy = 100 * correct / total
        print(f"\n🎯 Overall Test Accuracy: {overall_accuracy:.2f}%")
        
        print("\n📋 Per-class accuracy:")
        for i in range(10):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                print(f"   {class_names[i]:12}: {class_acc:.2f}%")
        
        return overall_accuracy
        
    def create_training_plot(self):
        """トレーニング進捗プロット作成"""
        print("📈 Creating training progress plot...")
        
        # チェックポイントから履歴取得
        checkpoint_files = list(self.recovery_system.checkpoint_dir.glob("nkat_checkpoint_*.pth"))
        
        if not checkpoint_files:
            print("⚠️ No checkpoint files found for plotting")
            return
        
        epochs = []
        train_accs = []
        val_accs = []
        losses = []
        
        for checkpoint_file in sorted(checkpoint_files, key=lambda x: x.stat().st_mtime):
            try:
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
                metrics = checkpoint.get('metrics', {})
                
                epochs.append(checkpoint.get('epoch', 0))
                train_accs.append(metrics.get('train_accuracy', 0))
                val_accs.append(metrics.get('val_accuracy', 0))
                losses.append(metrics.get('val_loss', 0))
                
            except Exception as e:
                continue
        
        if not epochs:
            print("⚠️ No valid checkpoint data for plotting")
            return
        
        # プロット作成
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Accuracy plot
        ax1.plot(epochs, train_accs, 'b-o', label='Train Accuracy', linewidth=2)
        ax1.plot(epochs, val_accs, 'r-o', label='Validation Accuracy', linewidth=2)
        ax1.axhline(y=60, color='green', linestyle='--', alpha=0.7, label='Target: 60%')
        ax1.set_title('NKAT CIFAR-10 Training Progress (Recovery Enabled)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(epochs, losses, 'g-', linewidth=2)
        ax2.set_title('Validation Loss', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_filename = f'nkat_cifar10_recovery_training_{timestamp}.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Training plot saved: {plot_filename}")

def main():
    """メイン実行関数"""
    print("🔋 NKAT CIFAR-10 Recovery Training System")
    print("="*60)
    
    # コマンドライン引数解析
    epochs = 50
    if len(sys.argv) > 1:
        try:
            epochs = int(sys.argv[1])
        except ValueError:
            print("⚠️ Invalid epoch number, using default: 50")
    
    # トレーナー初期化
    trainer = NKATCifarTrainer(
        batch_size=128,
        learning_rate=0.001,
        epochs=epochs,
        checkpoint_dir="checkpoints_cifar10_recovery"
    )
    
    # トレーニング実行
    best_accuracy = trainer.train_with_recovery()
    
    if best_accuracy is not None:
        # 最終評価
        final_accuracy = trainer.evaluate_model()
        
        # プロット作成
        trainer.create_training_plot()
        
        print(f"\n🎯 Final Results:")
        print(f"   Best Training Accuracy: {best_accuracy:.2f}%")
        print(f"   Final Test Accuracy: {final_accuracy:.2f}%")
        
        # 目標達成確認
        if final_accuracy >= 60:
            print("🎉 TARGET ACHIEVED! Accuracy ≥ 60%")
        elif final_accuracy >= 50:
            print("✅ Good progress! Close to target")
        else:
            print("📈 Continue training for better results")
    else:
        print("\n🔄 Training was interrupted. Run again to resume from checkpoint.")

if __name__ == "__main__":
    main() 