#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optuna ミニマルテスト
"""

print("🔬 Optuna Minimal Test Starting...")

try:
    import optuna
    print("✅ Optuna import successful")
    
    import torch
    print("✅ PyTorch import successful")
    
    from nkat_enhanced_cifar import NKATEnhancedConfig, NKATEnhancedTrainer
    print("✅ NKAT import successful")
    
    # Simple objective test
    def test_objective(trial):
        print(f"   Running trial {trial.number + 1}")
        
        # Simple parameter
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        
        print(f"   Suggested LR: {lr:.2e}")
        
        # Mock training result
        mock_accuracy = 50.0 + trial.number * 2  # Increasing accuracy
        print(f"   Mock accuracy: {mock_accuracy:.1f}%")
        
        return mock_accuracy
    
    # Create study
    print("\n📊 Creating Optuna study...")
    study = optuna.create_study(direction="maximize")
    
    # Run 3 simple trials
    print("🚀 Running 3 test trials...")
    study.optimize(test_objective, n_trials=3)
    
    print(f"\n✅ SUCCESS! Best value: {study.best_value:.1f}")
    print(f"Best params: {study.best_params}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 