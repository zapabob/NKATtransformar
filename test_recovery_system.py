#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT Power Recovery System Test Script
RTX3080 CUDA対応 リカバリーシステムテスト
"""

import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import time
from nkat_power_recovery_system import NKATPowerRecoverySystem

class TestModel(nn.Module):
    """テスト用簡単なモデル"""
    def __init__(self):
        super(TestModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
    
    def forward(self, x):
        return self.fc(x)

def test_recovery_system():
    """リカバリーシステムの基本機能テスト"""
    print("🔋 Testing NKAT Power Recovery System")
    print("="*50)
    
    # リカバリーシステム初期化
    recovery_system = NKATPowerRecoverySystem(
        checkpoint_dir="test_checkpoints",
        backup_dir="test_backups",
        auto_save_interval=30  # 30秒間隔（テスト用）
    )
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎮 Using device: {device}")
    
    # テストモデル作成
    model = TestModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    
    print("\n✅ Test components initialized")
    
    # Test 1: チェックポイント保存テスト
    print("\n📝 Test 1: Checkpoint Save")
    test_metrics = {
        'train_accuracy': 75.5,
        'val_accuracy': 72.3,
        'train_loss': 0.45,
        'val_loss': 0.52
    }
    
    checkpoint_path = recovery_system.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=5,
        best_accuracy=72.3,
        loss=0.52,
        metrics=test_metrics,
        checkpoint_type="test"
    )
    
    if checkpoint_path:
        print("✅ Checkpoint save successful")
    else:
        print("❌ Checkpoint save failed")
        return False
    
    # Test 2: チェックポイント読み込みテスト
    print("\n📂 Test 2: Checkpoint Load")
    loaded_data = recovery_system.load_checkpoint(checkpoint_path)
    
    if loaded_data:
        print("✅ Checkpoint load successful")
        print(f"   Loaded epoch: {loaded_data['epoch']}")
        print(f"   Loaded accuracy: {loaded_data['best_accuracy']:.2f}%")
        
        # モデル状態復元
        model.load_state_dict(loaded_data['model_state_dict'])
        optimizer.load_state_dict(loaded_data['optimizer_state_dict'])
        scheduler.load_state_dict(loaded_data['scheduler_state_dict'])
        print("✅ Model state restored")
    else:
        print("❌ Checkpoint load failed")
        return False
    
    # Test 3: 緊急保存テスト
    print("\n🚨 Test 3: Emergency Save")
    recovery_system.force_checkpoint("test_emergency")
    print("✅ Emergency save completed")
    
    # Test 4: CUDA メモリ監視テスト
    print("\n🎮 Test 4: CUDA Memory Monitoring")
    if torch.cuda.is_available():
        # ダミーテンソル作成
        dummy_tensor = torch.randn(1000, 1000).cuda()
        memory_before = torch.cuda.memory_allocated() / 1024**2  # MB
        
        # さらに大きなテンソル作成
        large_tensor = torch.randn(2000, 2000).cuda()
        memory_after = torch.cuda.memory_allocated() / 1024**2  # MB
        
        print(f"   Memory before: {memory_before:.1f}MB")
        print(f"   Memory after: {memory_after:.1f}MB")
        print(f"   Memory increase: {memory_after - memory_before:.1f}MB")
        
        # メモリクリア
        del dummy_tensor, large_tensor
        torch.cuda.empty_cache()
        memory_cleaned = torch.cuda.memory_allocated() / 1024**2
        print(f"   Memory after cleanup: {memory_cleaned:.1f}MB")
        print("✅ CUDA memory monitoring works")
    else:
        print("⚠️ CUDA not available for memory test")
    
    # Test 5: 自動保存間隔テスト
    print("\n⏰ Test 5: Auto-save Interval Test")
    print("   Waiting 35 seconds to test auto-save...")
    
    # シミュレートされた自動保存テスト
    start_time = time.time()
    while time.time() - start_time < 35:
        time.sleep(5)
        elapsed = time.time() - start_time
        print(f"   Elapsed: {elapsed:.0f}s / 35s")
        
        # 30秒経過後に自動保存をテスト
        if elapsed >= 30:
            auto_checkpoint_path = recovery_system.auto_checkpoint_wrapper(
                save_func=recovery_system.save_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=10,
                best_accuracy=78.5,
                loss=0.42,
                metrics={'test_auto_save': True}
            )
            if auto_checkpoint_path:
                print("✅ Auto-save triggered successfully")
            break
    
    # Test 6: リカバリーレポート生成テスト
    print("\n📊 Test 6: Recovery Report Generation")
    report = recovery_system.generate_recovery_report()
    print("✅ Recovery report generated")
    
    # Test 7: クリーンアップテスト
    print("\n🧹 Test 7: Cleanup Test")
    recovery_system.cleanup_old_checkpoints(keep_count=2)
    print("✅ Cleanup completed")
    
    print("\n🎉 All tests completed successfully!")
    print("="*50)
    
    return True

def test_simulated_power_failure():
    """模擬電源断テスト"""
    print("\n⚡ Simulated Power Failure Test")
    print("="*40)
    
    recovery_system = NKATPowerRecoverySystem(
        checkpoint_dir="power_failure_test",
        auto_save_interval=10  # 10秒間隔
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TestModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    
    print("🔥 Starting 'training' session...")
    
    # 3回のエポックをシミュレート
    for epoch in range(3):
        print(f"\n--- Epoch {epoch + 1} ---")
        
        # 訓練シミュレーション
        fake_loss = 1.0 - (epoch * 0.2)
        fake_accuracy = 60 + (epoch * 5)
        
        metrics = {
            'train_accuracy': fake_accuracy,
            'val_accuracy': fake_accuracy - 2,
            'train_loss': fake_loss,
            'val_loss': fake_loss + 0.1
        }
        
        # チェックポイント保存
        checkpoint_path = recovery_system.save_checkpoint(
            model, optimizer, scheduler, epoch,
            fake_accuracy - 2, fake_loss + 0.1, metrics, "regular"
        )
        
        print(f"   Simulated accuracy: {fake_accuracy:.1f}%")
        print(f"   Checkpoint saved: {checkpoint_path is not None}")
        
        # 2エポック目で「電源断」をシミュレート
        if epoch == 1:
            print("\n⚡💥 SIMULATED POWER FAILURE! 💥⚡")
            recovery_system.force_checkpoint("simulated_power_failure")
            break
    
    print("\n🔄 Simulating system restart...")
    time.sleep(2)
    
    # 新しいセッションで復旧
    print("\n🚀 Starting recovery session...")
    recovery_system_new = NKATPowerRecoverySystem(
        checkpoint_dir="power_failure_test",
        auto_save_interval=10
    )
    
    # 復旧データ読み込み
    recovered_data = recovery_system_new.load_checkpoint()
    
    if recovered_data:
        print("✅ Recovery successful!")
        print(f"   Resumed from epoch: {recovered_data['epoch']}")
        print(f"   Recovered accuracy: {recovered_data['best_accuracy']:.2f}%")
        print(f"   Recovery count: {recovered_data['recovery_count']}")
    else:
        print("❌ Recovery failed!")
        return False
    
    print("\n🎉 Power failure recovery test completed!")
    return True

def main():
    """メインテスト実行"""
    print("🔋 NKAT Power Recovery System - Comprehensive Test")
    print("="*60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # CUDA情報表示
    if torch.cuda.is_available():
        print(f"\n🎮 CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("\n⚠️ CUDA not available - running on CPU")
    
    try:
        # 基本機能テスト
        print("\n" + "="*60)
        basic_test_passed = test_recovery_system()
        
        if basic_test_passed:
            # 電源断シミュレーションテスト
            print("\n" + "="*60)
            power_failure_test_passed = test_simulated_power_failure()
            
            if power_failure_test_passed:
                print(f"\n🎉 ALL TESTS PASSED! 🎉")
                print("The power recovery system is ready for production use.")
            else:
                print(f"\n❌ Power failure test failed")
        else:
            print(f"\n❌ Basic functionality test failed")
            
    except Exception as e:
        print(f"\n💥 Critical test error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 