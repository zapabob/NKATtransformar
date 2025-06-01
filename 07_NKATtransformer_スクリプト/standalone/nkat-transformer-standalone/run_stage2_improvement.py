#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 Improvement Plan Execution
段階的 To-Do 実行スクリプト

実行手順：
1. CIFAR-10 Smoke Test
2. Optuna Optimization
3. Final Evaluation

Author: NKAT Advanced Computing Team
Version: 2.2.0 - Integrated Pipeline
"""

import sys
import os
import argparse
import json
from datetime import datetime

def check_dependencies():
    """依存関係チェック"""
    try:
        import torch
        import optuna
        import numpy as np
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        print("✅ All dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install: pip install -r requirements.txt")
        return False

def run_smoke_test():
    """🧪 Step 1: CIFAR-10 Smoke Test"""
    print("\n" + "="*50)
    print("🧪 Step 1: CIFAR-10 Smoke Test")
    print("="*50)
    
    try:
        from nkat_enhanced_cifar import quick_cifar_test
        success = quick_cifar_test()
        return success
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        return False

def run_optuna_optimization(n_trials=30):
    """🔍 Step 2: Optuna Optimization"""
    print("\n" + "="*50)  
    print(f"🔍 Step 2: Optuna Optimization ({n_trials} trials)")
    print("="*50)
    
    try:
        from optuna_stage2_optimization import run_optuna_optimization
        best_params, best_tpe = run_optuna_optimization(n_trials)
        return best_params, best_tpe
    except Exception as e:
        print(f"❌ Optuna optimization failed: {e}")
        return None, 0.0

def run_final_evaluation(best_params):
    """📊 Step 3: Final Multi-Dataset Evaluation"""
    print("\n" + "="*50)
    print("📊 Step 3: Final Multi-Dataset Evaluation")
    print("="*50)
    
    datasets = ['mnist', 'fashion', 'cifar10']
    results = {}
    
    try:
        from nkat_enhanced_cifar import NKATEnhancedConfig, NKATEnhancedTrainer
        
        for dataset in datasets:
            print(f"\n🎯 Training on {dataset.upper()}...")
            
            # Create optimized config
            config = NKATEnhancedConfig(dataset)
            
            # Apply best parameters
            for key, value in best_params.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            # Full training for final evaluation
            config.num_epochs = 40 if dataset == 'cifar10' else 30
            config.batch_size = 32
            
            # Train
            trainer = NKATEnhancedTrainer(config)
            accuracy, tpe = trainer.train()
            
            # Store results
            total_params = sum(p.numel() for p in trainer.model.parameters())
            results[dataset] = {
                'accuracy': accuracy,
                'tpe': tpe,
                'params': total_params,
                'lambda_theory': total_params / 1e6
            }
            
            print(f"{dataset.upper()}: Acc={accuracy:.2f}%, TPE={tpe:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Final evaluation failed: {e}")
        return {}

def calculate_stage2_metrics(results):
    """Stage 2 最終メトリクス計算"""
    if not results:
        return {}
    
    import numpy as np
    import math
    
    # Individual TPEs
    tpes = [r['tpe'] for r in results.values()]
    accuracies = [r['accuracy'] for r in results.values()]
    
    # Global TPE
    avg_accuracy = np.mean(accuracies)
    avg_lambda = np.mean([r['lambda_theory'] for r in results.values()])
    global_tpe = (avg_accuracy / 100.0) / math.log10(1 + avg_lambda)
    
    # Consistency (standard deviation of TPEs)
    consistency = 1.0 - (np.std(tpes) / np.mean(tpes)) if np.mean(tpes) > 0 else 0.0
    
    # Robustness (minimum TPE ratio to mean)
    robustness = min(tpes) / np.mean(tpes) if np.mean(tpes) > 0 else 0.0
    
    return {
        'global_tpe': global_tpe,
        'consistency': consistency,
        'robustness': robustness,
        'individual_tpes': tpes,
        'individual_accuracies': accuracies
    }

def save_final_report(results, metrics, best_params):
    """最終レポート保存"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = {
        'timestamp': timestamp,
        'stage': 'Stage 2 Enhanced',
        'results': results,
        'metrics': metrics,
        'best_parameters': best_params,
        'summary': {
            'global_tpe': metrics.get('global_tpe', 0.0),
            'target_achieved': metrics.get('global_tpe', 0.0) >= 0.70,
            'consistency_ok': metrics.get('consistency', 0.0) >= 0.80,
            'robustness_ok': metrics.get('robustness', 0.0) >= 0.75
        }
    }
    
    filename = f"stage2_final_report_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Final report saved: {filename}")
    return filename

def print_final_summary(metrics, results):
    """最終サマリー表示"""
    print("\n" + "="*60)
    print("🎉 STAGE 2 FINAL RESULTS")
    print("="*60)
    
    # Target table
    print("\n📊 目標達成状況:")
    print("| 指標              | 現状       | 目標     | ステータス    |")
    print("| --------------- | -------- | ------ | --------- |")
    
    global_tpe = metrics.get('global_tpe', 0.0)
    consistency = metrics.get('consistency', 0.0)  
    robustness = metrics.get('robustness', 0.0)
    
    tpe_status = "✅ OK" if global_tpe >= 0.70 else "❌ NG"
    con_status = "✅ OK" if consistency >= 0.80 else "❌ NG"
    rob_status = "✅ OK" if robustness >= 0.75 else "❌ NG"
    
    print(f"| **Global TPE**  | **{global_tpe:.2f}** | ≥ 0.70 | **{tpe_status}** |")
    print(f"| **Consistency** | {consistency:.2f}     | ≥ 0.80 | {con_status}     |")
    print(f"| **Robustness**  | {robustness:.2f}     | ≥ 0.75 | {rob_status}     |")
    
    # Individual results
    print("\n📈 Individual Dataset Results:")
    for dataset, result in results.items():
        print(f"  {dataset.upper()}: Acc={result['accuracy']:.2f}%, TPE={result['tpe']:.4f}")
    
    # Overall assessment
    print("\n🎯 Overall Assessment:")
    if global_tpe >= 0.70:
        print("🎉 **目標達成！** Stage 2 成功")
        print("✨ 薄めの理論で広い世界を制覇するバランス達成")
    else:
        print(f"📈 継続改善が必要 (目標: ≥ 0.70, 現在: {global_tpe:.2f})")
        print("🔧 追加調整を推奨")

def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description='Stage 2 Improvement Plan Execution')
    parser.add_argument('--trials', type=int, default=30, help='Optuna trials (default: 30)')
    parser.add_argument('--skip-smoke', action='store_true', help='Skip smoke test')
    parser.add_argument('--skip-optuna', action='store_true', help='Skip Optuna optimization')
    parser.add_argument('--quick', action='store_true', help='Quick mode (reduced trials)')
    
    args = parser.parse_args()
    
    if args.quick:
        args.trials = 10
    
    print("🚀 NKAT Stage 2 Improvement Plan")
    print("CIFAR-10対応＆Global TPE最大化")
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Step 1: Smoke Test
    if not args.skip_smoke:
        success = run_smoke_test()
        if not success:
            print("❌ Smoke test failed. Aborting.")
            sys.exit(1)
        print("✅ Smoke test passed! Proceeding to optimization...")
    
    # Step 2: Optuna Optimization  
    if not args.skip_optuna:
        best_params, best_tpe = run_optuna_optimization(args.trials)
        if best_params is None:
            print("❌ Optuna optimization failed. Using default parameters.")
            # Use enhanced default parameters
            from nkat_enhanced_cifar import NKATEnhancedConfig
            config = NKATEnhancedConfig('cifar10')
            best_params = {
                'patch_size': config.patch_size,
                'conv_stem': config.conv_stem,
                'embed_dim': config.embed_dim,
                'depth': config.depth,
                'temperature': config.temperature,
                'top_k': config.top_k,
                'top_p': config.top_p,
                'nkat_strength': config.nkat_strength,
                'dropout_attn': config.dropout_attn,
                'dropout_embed': config.dropout_embed,
                'learning_rate': config.learning_rate,
                'mixup_alpha': config.mixup_alpha,
                'cutmix_prob': config.cutmix_prob,
            }
    else:
        # Use default enhanced parameters
        from nkat_enhanced_cifar import NKATEnhancedConfig
        config = NKATEnhancedConfig('cifar10')
        best_params = {
            'patch_size': config.patch_size,
            'conv_stem': config.conv_stem,
            'embed_dim': config.embed_dim,
            'depth': config.depth,
            'temperature': config.temperature,
            'top_k': config.top_k,
            'top_p': config.top_p,
            'nkat_strength': config.nkat_strength,
            'dropout_attn': config.dropout_attn,
            'dropout_embed': config.dropout_embed,
            'learning_rate': config.learning_rate,
            'mixup_alpha': config.mixup_alpha,
            'cutmix_prob': config.cutmix_prob,
        }
    
    # Step 3: Final Evaluation
    results = run_final_evaluation(best_params)
    
    # Calculate metrics
    metrics = calculate_stage2_metrics(results)
    
    # Save report
    save_final_report(results, metrics, best_params)
    
    # Print summary
    print_final_summary(metrics, results)

if __name__ == "__main__":
    main() 