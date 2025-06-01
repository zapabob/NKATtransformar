#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT-Transformer Stage 2: CIFAR-10 Smoke Test (10 epochs)
Goal: Val Acc ≥ 45%
"""

if __name__ == "__main__":
    try:
        print("🚀 NKAT Stage 2 CIFAR-10 Smoke Test Starting...")
        print("=" * 60)
        
        from nkat_enhanced_cifar import NKATEnhancedConfig, NKATEnhancedTrainer
        
        # Configuration for 10-epoch smoke test
        config = NKATEnhancedConfig('cifar10')
        config.num_epochs = 10
        config.batch_size = 32
        config.learning_rate = 1e-4
        config.weight_decay = 1e-4
        config.use_cuda = True
        
        print(f"📋 Config: epochs={config.num_epochs}, batch_size={config.batch_size}")
        print(f"🎯 Target: Val Acc ≥ 45%")
        print("=" * 60)
        
        # Initialize trainer and run
        trainer = NKATEnhancedTrainer(config)
        accuracy, tpe = trainer.train()
        
        print("=" * 60)
        print(f"🏁 Final Results:")
        print(f"   Validation Accuracy: {accuracy:.2f}%")
        print(f"   TPE Score: {tpe:.4f}")
        
        if accuracy >= 45.0:
            print("✅ SMOKE TEST PASSED! Ready for Optuna optimization!")
        else:
            print("⚠️  Below target. Need parameter tuning.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc() 