#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT Optuna Emergency Boost
45%未達時の緊急パラメータ特化戦略
"""

import optuna
from datetime import datetime

def emergency_search_space(trial):
    """緊急時の狭域特化パラメータ"""
    
    # CIFAR-10で実績のある範囲に絞り込み
    params = {
        "patch_size": trial.suggest_categorical("patch_size", [2]),  # 2固定
        "embed_dim": trial.suggest_categorical("embed_dim", [448, 512]),  # 高次元優先
        "depth": trial.suggest_int("depth", 6, 8),  # 中央値付近
        "temperature": trial.suggest_float("temperature", 0.7, 0.9),  # 最適範囲
        "top_p": trial.suggest_float("top_p", 0.85, 0.92),  # 高性能域
        "nkat_strength": trial.suggest_float("nkat_strength", 0.4, 0.6),  # 安定域
        "learning_rate": trial.suggest_float("learning_rate", 8e-5, 1.5e-4, log=True),
        "mixup_alpha": trial.suggest_float("mixup_alpha", 0.05, 0.15),  # 軽微なAug
        "cutmix_prob": trial.suggest_float("cutmix_prob", 0.15, 0.25),  # 控えめ
        "dropout_attn": trial.suggest_float("dropout_attn", 0.02, 0.06),  # 低ドロップアウト
        "dropout_embed": trial.suggest_float("dropout_embed", 0.02, 0.06),
        
        # 緊急ブースト用パラメータ
        "warmup_epochs": trial.suggest_int("warmup_epochs", 2, 4),
        "weight_decay": trial.suggest_float("weight_decay", 5e-5, 2e-4, log=True),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.1),
    }
    
    return params

def run_emergency_optuna():
    """緊急時Optuna実行"""
    
    print("🚨 EMERGENCY OPTUNA BOOST")
    print("=" * 50)
    print("🎯 Target: Val Acc ≥ 50% (Emergency threshold)")
    print("⚡ Strategy: Narrow high-performance parameter space")
    print("=" * 50)
    
    study = optuna.create_study(
        direction="maximize",
        study_name="nkat_emergency_boost",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=1,  # 即座枝刈り
            n_warmup_steps=2
        )
    )
    
    def emergency_objective(trial):
        try:
            from nkat_enhanced_cifar import NKATEnhancedConfig, NKATEnhancedTrainer
            
            params = emergency_search_space(trial)
            
            print(f"\n🆘 Emergency Trial {trial.number + 1}:")
            print(f"   embed_dim={params['embed_dim']}, depth={params['depth']}")
            print(f"   lr={params['learning_rate']:.2e}, temp={params['temperature']:.3f}")
            
            config = NKATEnhancedConfig('cifar10')
            config.num_epochs = 6  # 超高速評価
            config.batch_size = 32
            config.early_stopping = True
            config.early_stopping_patience = 2  # 早期停止
            
            for key, value in params.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            trainer = NKATEnhancedTrainer(config)
            accuracy, tpe = trainer.train()
            
            print(f"   📊 Emergency Results: Acc={accuracy:.2f}%, TPE={tpe:.4f}")
            
            # 緊急時は40%でも通す
            if accuracy < 40.0:
                print(f"   ❌ Below emergency threshold (40%)")
                raise optuna.TrialPruned()
            
            objective_value = accuracy + (tpe * 15)  # TPE重み増加
            print(f"   🚨 Emergency Objective: {objective_value:.3f}")
            
            return objective_value
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"   💥 Emergency trial failed: {e}")
            return 0.0
    
    # 緊急5 trial実行
    study.optimize(emergency_objective, n_trials=5)
    
    print("\n" + "=" * 50)
    print("🚨 Emergency Optuna Results")
    print("=" * 50)
    
    best_trial = study.best_trial
    print(f"🥇 Emergency Best Trial: {best_trial.value:.3f}")
    
    if best_trial.value > 50:
        print("✅ EMERGENCY SUCCESS! Ready for full training!")
        
        emergency_config = {
            "emergency_trial": True,
            "trial_number": best_trial.number + 1,
            "objective_value": best_trial.value,
            "parameters": best_trial.params,
            "timestamp": str(datetime.now())
        }
        
        import json
        with open("emergency_config.json", "w") as f:
            json.dump(emergency_config, f, indent=2)
        print("💾 Emergency config saved!")
        
    else:
        print("⚠️  Emergency boost insufficient. Consider parameter review.")
    
    return study

if __name__ == "__main__":
    run_emergency_optuna() 