#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Test Script for NKAT Enhanced CIFAR
動作確認用クイックテスト

段階別テスト：
1. モジュールインポート確認
2. モデル作成確認  
3. データローダー確認
4. 1エポック訓練確認

Author: NKAT Advanced Computing Team
"""

import torch
import sys
import traceback

def test_imports():
    """インポートテスト"""
    print("🔍 Testing imports...")
    try:
        from nkat_enhanced_cifar import (
            NKATEnhancedConfig, 
            NKATEnhancedViT, 
            NKATEnhancedTrainer,
            create_enhanced_dataloaders
        )
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_model_creation():
    """モデル作成テスト"""
    print("\n🔍 Testing model creation...")
    try:
        from nkat_enhanced_cifar import NKATEnhancedConfig, NKATEnhancedViT
        
        config = NKATEnhancedConfig('cifar10')
        model = NKATEnhancedViT(config)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created successfully")
        print(f"   Parameters: {total_params:,}")
        print(f"   λ_theory: {total_params/1e6:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        traceback.print_exc()
        return False

def test_dataloader():
    """データローダーテスト"""
    print("\n🔍 Testing dataloader...")
    try:
        from nkat_enhanced_cifar import NKATEnhancedConfig, create_enhanced_dataloaders
        
        config = NKATEnhancedConfig('cifar10')
        config.batch_size = 8  # Small batch for testing
        
        train_loader, test_loader = create_enhanced_dataloaders(config)
        
        # Test one batch
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"✅ Dataloader working")
            print(f"   Batch shape: {data.shape}")
            print(f"   Target shape: {target.shape}")
            print(f"   Data range: [{data.min():.3f}, {data.max():.3f}]")
            break
            
        return True
    except Exception as e:
        print(f"❌ Dataloader failed: {e}")
        traceback.print_exc()
        return False

def test_forward_pass():
    """フォワードパステスト"""
    print("\n🔍 Testing forward pass...")
    try:
        from nkat_enhanced_cifar import NKATEnhancedConfig, NKATEnhancedViT
        
        config = NKATEnhancedConfig('cifar10')
        model = NKATEnhancedViT(config)
        
        # Dummy input
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 32, 32)
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
            
        print(f"✅ Forward pass successful")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        traceback.print_exc()
        return False

def test_one_epoch():
    """1エポック訓練テスト"""
    print("\n🔍 Testing one epoch training...")
    try:
        from nkat_enhanced_cifar import NKATEnhancedConfig, NKATEnhancedTrainer
        
        config = NKATEnhancedConfig('cifar10')
        config.num_epochs = 1
        config.batch_size = 16
        
        trainer = NKATEnhancedTrainer(config)
        
        # Limit to few batches for quick test
        original_train_loader = trainer.train_loader
        limited_data = []
        for i, batch in enumerate(original_train_loader):
            limited_data.append(batch)
            if i >= 5:  # Only 5 batches
                break
        
        print("🚀 Starting 1-epoch test...")
        
        trainer.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(limited_data):
            data, target = data.to(trainer.device), target.to(trainer.device)
            
            trainer.optimizer.zero_grad()
            output = trainer.model(data)
            loss = trainer.criterion(output, target)
            loss.backward()
            trainer.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            print(f"   Batch {batch_idx+1}: Loss={loss.item():.4f}")
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(limited_data)
        
        print(f"✅ One epoch test successful")
        print(f"   Accuracy: {accuracy:.2f}%")
        print(f"   Average Loss: {avg_loss:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ One epoch test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("🚀 NKAT Enhanced CIFAR Quick Test")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Dataloader", test_dataloader),
        ("Forward Pass", test_forward_pass),
        ("One Epoch", test_one_epoch),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
            if not success:
                print(f"⚠️  {test_name} failed, stopping tests")
                break
        except KeyboardInterrupt:
            print(f"🛑 Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
            break
    
    print("\n" + "="*50)
    print("📊 Test Results Summary")
    print("="*50)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:15}: {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 All tests passed! Ready for CIFAR-10 training")
    else:
        print("\n🔧 Some tests failed. Please check the errors above")
    
    return all_passed

if __name__ == "__main__":
    main() 