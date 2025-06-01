#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple CIFAR-10 Test with Error Handling
エラー回避版シンプルCIFARテスト
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import shutil

def safe_cifar_dataloader(batch_size=32):
    """Safe CIFAR-10 dataloader with error handling"""
    
    # Simple transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Try to use existing data
    try:
        print("🔍 Trying to use existing CIFAR-10 data...")
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=False, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=False, transform=transform
        )
        print("✅ Existing data loaded successfully")
        
    except Exception as e:
        print(f"⚠️  Existing data failed: {e}")
        print("🧹 Cleaning data directory...")
        
        # Clean up corrupted data
        try:
            if os.path.exists('./data/cifar-10-batches-py'):
                shutil.rmtree('./data/cifar-10-batches-py')
            if os.path.exists('./data/cifar-10-python.tar.gz'):
                os.remove('./data/cifar-10-python.tar.gz')
        except Exception as cleanup_error:
            print(f"Clean up warning: {cleanup_error}")
        
        print("🌐 Re-downloading CIFAR-10...")
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        print("✅ Fresh data downloaded successfully")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, test_loader

def simple_model_test():
    """Simple model functionality test"""
    print("🧪 Simple CIFAR-10 Model Test")
    print("="*40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")
    
    # Test data loading
    try:
        train_loader, test_loader = safe_cifar_dataloader(batch_size=16)
        
        # Test one batch
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"✅ Data batch loaded: {data.shape}")
            break
            
        # Test enhanced model creation
        from nkat_enhanced_cifar import NKATEnhancedConfig, NKATEnhancedViT
        
        config = NKATEnhancedConfig('cifar10')
        model = NKATEnhancedViT(config).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created: {total_params:,} parameters")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            data = data.to(device)
            output = model(data)
            print(f"✅ Forward pass: {output.shape}")
        
        # Quick training test
        print("\n🚀 Quick training test (5 batches)...")
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 5:  # Only 5 batches
                break
                
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            print(f"   Batch {batch_idx+1}: Loss={loss.item():.4f}")
        
        avg_loss = total_loss / 5
        print(f"\n✅ Training test successful! Avg Loss: {avg_loss:.4f}")
        
        # Calculate TPE estimate
        lambda_theory = total_params / 1e6
        estimated_accuracy = 30.0  # Conservative estimate for 5 batches
        tpe_estimate = estimated_accuracy / 100.0 / torch.log10(torch.tensor(1 + lambda_theory))
        
        print(f"\n📊 Performance Estimates:")
        print(f"   λ_theory: {lambda_theory:.3f}")
        print(f"   Estimated Accuracy: {estimated_accuracy:.1f}%")
        print(f"   Estimated TPE: {tpe_estimate:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_model_test()
    if success:
        print("\n🎉 Simple test passed! Enhanced CIFAR model is working")
        print("✨ Ready for full smoke test and optimization")
    else:
        print("\n❌ Simple test failed. Please check the errors above") 