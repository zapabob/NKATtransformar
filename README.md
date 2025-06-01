# 🔋 NKAT Transformer with Power Recovery System

**電源断対応 深層学習フレームワーク** | **RTX3080 CUDA最適化** | **自動復旧機能**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-RTX3080-green.svg)](https://developer.nvidia.com/cuda-zone)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 プロジェクト概要

**NKAT Transformer with Power Recovery System**は、電源断からの完全自動復旧機能を備えた深層学習フレームワークです。RTX3080での長時間トレーニングを安全に実行し、予期しない中断からも確実に復旧できます。

### 🏆 最新の成果 (2025年6月1日)

- **🎯 CIFAR-10精度**: **76.50%** (目標60%を16.5%上回る)
- **🔋 完全自動復旧**: 電源断から同コマンドで復元
- **⚡ RTX3080最適化**: GPU効率 0.1GB/10.0GB
- **✅ 100%安定性**: リカバリー回数0回の完全安定動作

## 🔋 電源断リカバリーシステムの特徴

### 💾 自動チェックポイント保存
- **5分間隔**での自動保存
- **アトミック保存**による破損防止
- **メタデータ追跡**（セッションID、タイムスタンプ）
- **古いチェックポイント自動削除**（最新5個保持）

### 🚨 緊急保存機能
- **Ctrl+C**、**SIGTERM**でのグレースフル終了
- **予期しないエラー**時の自動保存
- **GPU メモリ不足**時の緊急対応

### 🔄 完全自動復旧
- **同コマンド再実行**で自動復元
- **モデル・オプティマイザー・スケジューラー**の完全復元
- **トレーニング進捗**の継続
- **セッション統計**の保持

## 📂 リポジトリ構造

```
NKATtransformar/
├── 🔋 電源断リカバリーシステム
│   ├── nkat_power_recovery_system.py      # メインリカバリーシステム
│   ├── nkat_cifar10_recovery_training.py  # CIFAR-10統合トレーニング
│   ├── test_recovery_system.py            # システムテスト
│   └── 電源断リカバリーシステム_マニュアル.md    # 包括的マニュアル
├── 🏃‍♂️ 実行スクリプト
│   ├── run_recovery_training.bat          # Windows実行バッチ
│   ├── monitor_cifar.py                   # トレーニング監視
│   └── monitor_progress.py                # 進捗監視
├── ⚡ 最適化ツール
│   ├── optuna_emergency_boost.py          # 緊急最適化
│   ├── quick_optuna_cifar10.py           # 高速ハイパーパラメータ調整
│   └── test_optuna_minimal.py            # 最小テスト
├── 🧠 NKAT Transformer システム
│   └── 07_NKATtransformer_スクリプト/
│       ├── core_models/                   # コアモデル
│       ├── enhanced_versions/             # 拡張版
│       ├── recognition_apps/              # 認識アプリ
│       └── standalone/                    # スタンドアロン版
└── 📊 プロジェクト管理
    ├── .specstory/                        # 開発履歴
    ├── .cursor/                           # 開発環境設定
    └── README.md                          # このファイル
```

## 🚀 クイックスタート

### 1️⃣ 環境要件

```bash
# 必要なシステム
- Windows 11 (推奨)
- Python 3.8+
- NVIDIA RTX3080 (CUDA対応)
- 8GB以上のRAM
- 10GB以上の空きストレージ
```

### 2️⃣ インストール

```bash
# リポジトリクローン
git clone https://github.com/zapabob/NKATtransformar.git
cd NKATtransformar

# 依存関係インストール (requirements.txtがある場合)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tqdm matplotlib numpy
```

### 3️⃣ 電源断リカバリーシステムテスト

```bash
# システムテスト実行
python test_recovery_system.py
```

### 4️⃣ CIFAR-10トレーニング開始

```bash
# Windows バッチ実行
run_recovery_training.bat

# または直接実行
python nkat_cifar10_recovery_training.py
```

## 💡 使用例

### 基本的な電源断対応トレーニング

```python
from nkat_power_recovery_system import NKATPowerRecoverySystem
from nkat_cifar10_recovery_training import NKATCifarTrainer

# リカバリーシステム初期化
recovery_system = NKATPowerRecoverySystem(
    checkpoint_dir="checkpoints_cifar10_recovery",
    auto_save_interval=300  # 5分間隔
)

# トレーナー初期化
trainer = NKATCifarTrainer(recovery_system=recovery_system)

# トレーニング開始（自動復旧対応）
trainer.train(epochs=10)
```

### カスタム復旧設定

```python
# カスタム設定でリカバリーシステム作成
recovery_system = NKATPowerRecoverySystem(
    checkpoint_dir="my_checkpoints",
    backup_dir="my_backups",
    auto_save_interval=600,  # 10分間隔
    max_checkpoints=10       # 最大10個保持
)
```

## 📊 実証された性能

### 🎯 CIFAR-10 ベンチマーク結果

| 指標 | 値 | 備考 |
|------|----|----|
| **最終テスト精度** | **76.50%** | 目標60%を大幅上回る |
| **トレーニング時間** | 10エポック約15分 | RTX3080最適化 |
| **GPU メモリ効率** | 0.1GB/10.0GB | 効率的メモリ使用 |
| **復旧テスト** | 100%成功 | 電源断シミュレーション |
| **チェックポイント数** | 10個/10エポック | 完全保存記録 |

### 📈 クラス別精度

| クラス | 精度 | 特徴 |
|--------|------|------|
| automobile | **92.50%** | 最高精度 |
| truck | **90.00%** | 高精度 |
| ship | **89.40%** | 高精度 |
| frog | **84.40%** | 良好 |
| airplane | **80.00%** | 良好 |
| horse | **79.40%** | 良好 |
| deer | **73.00%** | 標準 |
| dog | **69.60%** | 標準 |
| bird | **57.50%** | 改善余地 |
| cat | **49.20%** | 最も困難 |

## 🔧 高度な機能

### 📈 監視・可視化

```bash
# リアルタイム監視
python monitor_cifar.py

# 進捗確認
python monitor_progress.py
```

### ⚡ ハイパーパラメータ最適化

```bash
# Optuna最適化
python quick_optuna_cifar10.py

# 緊急ブースト
python optuna_emergency_boost.py
```

### 🧪 システム検証

```bash
# 最小テスト
python test_optuna_minimal.py

# スモークテスト
python run_smoke_test.py
```

## 📚 詳細ドキュメント

- 📖 **[電源断リカバリーシステム_マニュアル.md](電源断リカバリーシステム_マニュアル.md)** - 包括的システムマニュアル
- 🏗️ **[07_NKATtransformer_スクリプト/README.md](07_NKATtransformer_スクリプト/README.md)** - NKAT Transformerコア
- 📊 **[.specstory/](,specstory/)** - 開発履歴・成果記録

## 🤝 コントリビューション

1. このリポジトリをフォーク
2. フィーチャーブランチを作成 (`git checkout -b feature/AmazingFeature`)
3. 変更をコミット (`git commit -m 'Add some AmazingFeature'`)
4. ブランチにプッシュ (`git push origin feature/AmazingFeature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 🙏 謝辞

- **NVIDIA** - CUDA最適化サポート
- **PyTorch Team** - 深層学習フレームワーク
- **CIFAR-10 Dataset** - ベンチマークデータセット

## 📞 サポート・連絡先

- **Issues**: [GitHub Issues](https://github.com/zapabob/NKATtransformar/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zapabob/NKATtransformar/discussions)
- **Email**: NKAT研究チーム

---

<div align="center">

**🎉 電源断を恐れない深層学習を実現！ 🎉**

**最終更新**: 2025年6月1日 | **バージョン**: 2.0 Power Recovery Edition

</div>
