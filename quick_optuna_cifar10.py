#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT Stage 2: CIFAR-10特化 クイックOptuna最適化
Goal: 10 trial で Val ≥ 55% を発見

探索空間: CIFAR特化
フィルタ: Val < 45% は即座にPrune
"""

import optuna
import torch
import numpy as np
import math
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def objective(trial):
    """CIFAR-10特化目的関数"""
    
    # === CIFAR特化探索空間 ===
    params = {
        "patch_size": trial.suggest_categorical("patch_size", [2, 4]),
        "embed_dim": trial.suggest_int("embed_dim", 384, 512, step=64),
        "depth": trial.suggest_int("depth", 5, 9),
        "temperature": trial.suggest_float("temperature", 0.6, 1.0),
        "top_p": trial.suggest_float("top_p", 0.80, 0.95),
        "nkat_strength": trial.suggest_float("nkat_strength", 0.3, 0.7),
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 2e-4, log=True),
        "mixup_alpha": trial.suggest_float("mixup_alpha", 0.0, 0.2),
        "cutmix_prob": trial.suggest_float("cutmix_prob", 0.1, 0.3),
        "dropout_attn": trial.suggest_float("dropout_attn", 0.0, 0.1),
        "dropout_embed": trial.suggest_float("dropout_embed", 0.0, 0.1),
    }
    
    print(f"\n🔬 Trial {trial.number + 1}/10:")
    print(f"   patch_size={params['patch_size']}, embed_dim={params['embed_dim']}")
    print(f"   depth={params['depth']}, temp={params['temperature']:.3f}")
    
    try:
        from nkat_enhanced_cifar import NKATEnhancedConfig, NKATEnhancedTrainer
        
        # Configuration
        config = NKATEnhancedConfig('cifar10')
        config.num_epochs = 8  # Quick evaluation
        config.batch_size = 32
        config.early_stopping = True
        config.early_stopping_patience = 3
        
        # Apply Optuna parameters
        for key, value in params.items():
            setattr(config, key, value)
        
        # Quick training
        trainer = NKATEnhancedTrainer(config)
        accuracy, tpe = trainer.train()
        
        print(f"   📊 Results: Acc={accuracy:.2f}%, TPE={tpe:.4f}")
        
        # Early filtering: 45%未満は即座にPrune
        if accuracy < 45.0:
            print(f"   ❌ Below 45% threshold. Pruning trial.")
            raise optuna.TrialPruned()
        
        # TPE-weighted objective (accuracy優先 but TPE bonus)
        objective_value = accuracy + (tpe * 10)  # TPE boost
        
        print(f"   ✅ Objective: {objective_value:.3f}")
        
        return objective_value
        
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"   💥 Trial failed: {e}")
        return 0.0

def run_quick_optuna():
    """クイック10 trial Optuna実行"""
    
    print("🚀 NKAT Stage 2: CIFAR-10 Quick Optuna (10 trials)")
    print("=" * 60)
    print("🎯 Goal: Find Val Acc ≥ 55% configuration")
    print("⚡ Strategy: Early pruning + TPE-weighted optimization")
    print("=" * 60)
    
    # Study作成
    study = optuna.create_study(
        direction="maximize",
        study_name="nkat_cifar10_quick",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=2,
            n_warmup_steps=3
        )
    )
    
    # 10 trial実行
    study.optimize(objective, n_trials=10, show_progress_bar=True)
    
    print("\n" + "=" * 60)
    print("🏁 Quick Optuna Results")
    print("=" * 60)
    
    # Best trial
    best_trial = study.best_trial
    print(f"🥇 Best Trial #{best_trial.number + 1}:")
    print(f"   Objective Value: {best_trial.value:.3f}")
    
    print(f"\n📋 Best Parameters:")
    for key, value in best_trial.params.items():
        print(f"   {key}: {value}")
    
    # フル学習推奨チェック
    estimated_full_acc = best_trial.value * 0.8  # 8 epoch → 40 epoch推定
    if estimated_full_acc >= 55.0:
        print(f"\n✅ SUCCESS: Estimated 40-epoch Acc ≈ {estimated_full_acc:.1f}%")
        print("🚀 Ready for full 40-epoch training!")
        
        # Save best config
        best_config = {
            "trial_number": best_trial.number + 1,
            "objective_value": best_trial.value,
            "estimated_full_accuracy": estimated_full_acc,
            "parameters": best_trial.params,
            "timestamp": str(datetime.now())
        }
        
        import json
        with open("best_optuna_config.json", "w") as f:
            json.dump(best_config, f, indent=2)
        print("💾 Saved to best_optuna_config.json")
        
    else:
        print(f"\n⚠️  Need more optimization. Best estimated: {estimated_full_acc:.1f}%")
        print("🔄 Consider running more trials or adjusting search space.")
    
    print("=" * 60)
    
    return study

if __name__ == "__main__":
    try:
        study = run_quick_optuna()
    except KeyboardInterrupt:
        print("\n⏸️  Optuna interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc() 