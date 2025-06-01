#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔋 NKAT with Non-Commutative Kolmogorov-Arnold Representation Theory
非可換コルモゴロフアーノルド表現理論統合版

Mathematical Foundation:
- Non-commutative KA representation: f(x) = Σ φᵢ(x) ⊗ ψᵢ(Aᵢx)
- Multi-resolution patch embedding with KA operators
- Temperature-controlled attention with KA enhancements
- Power recovery system integration

Features:
- 非可換KA表現による関数近似
- 多解像度パッチ (2×2 & 4×4) + KA演算子
- 温度制御アテンションのKA拡張
- Focal Loss + EMA + 電源断リカバリー
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import uuid
from datetime import datetime
import signal
import threading
import json
import math

# 電源断リカバリーシステム読み込み
try:
    from nkat_power_recovery_system import NKATPowerRecoverySystem
except ImportError:
    print("⚠️ Power recovery system not found, using basic mode")
    NKATPowerRecoverySystem = None

class NonCommutativeKALayer(nn.Module):
    """
    非可換コルモゴロフアーノルド表現層
    
    数学的基盤:
    f(x) = Σᵢ φᵢ(x) ⊗ ψᵢ(Aᵢx)
    
    where:
    - φᵢ: univariate activation functions
    - ψᵢ: multivariate basis functions  
    - Aᵢ: non-commutative transformation matrices
    - ⊗: tensor product operation
    """
    
    def __init__(self, input_dim, output_dim, num_ka_terms=8, temperature=0.72):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_ka_terms = num_ka_terms
        self.temperature = temperature
        
        # 非可換変換行列 Aᵢ
        self.A_matrices = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, input_dim) / math.sqrt(input_dim))
            for _ in range(num_ka_terms)
        ])
        
        # φᵢ: 単変数活性化関数群
        self.phi_functions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.SiLU(),
                nn.Linear(input_dim // 2, output_dim // num_ka_terms)
            ) for _ in range(num_ka_terms)
        ])
        
        # ψᵢ: 多変数基底関数群
        self.psi_functions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.GELU(),
                nn.Linear(input_dim, output_dim // num_ka_terms)
            ) for _ in range(num_ka_terms)
        ])
        
        # 非可換混合重み
        self.mixing_weights = nn.Parameter(torch.ones(num_ka_terms) / num_ka_terms)
        
        # 温度制御パラメータ
        self.temp_scale = nn.Parameter(torch.tensor(temperature))
        
    def forward(self, x):
        """
        非可換KA表現の前向き計算
        """
        batch_size = x.size(0)
        ka_terms = []
        
        for i in range(self.num_ka_terms):
            # 非可換変換: Aᵢx
            A_x = torch.matmul(x, self.A_matrices[i])
            
            # φᵢ(x): 単変数関数適用
            phi_x = self.phi_functions[i](x)
            
            # ψᵢ(Aᵢx): 多変数基底関数適用
            psi_Ax = self.psi_functions[i](A_x)
            
            # テンソル積操作（簡略化版）
            # 実際のテンソル積の代わりに要素積+温度スケーリング
            ka_term = phi_x * psi_Ax * self.temp_scale
            ka_terms.append(ka_term)
        
        # 非可換重み付き和
        weighted_terms = []
        for i, term in enumerate(ka_terms):
            weighted_terms.append(self.mixing_weights[i] * term)
        
        # 最終結合
        output = torch.cat(weighted_terms, dim=-1)
        
        return output

class MultiResKAPatchEmbedding(nn.Module):
    """KA表現を統合した多解像度パッチ埋め込み"""
    
    def __init__(self, img_size=32, patch_sizes=(2, 4), in_chans=3, embed_dim=448):
        super().__init__()
        self.img_size = img_size
        self.patch_sizes = patch_sizes
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        # 各解像度用のパッチ埋め込み
        self.patch_embeds = nn.ModuleDict()
        self.ka_layers = nn.ModuleDict()
        self.num_patches = {}
        
        for patch_size in patch_sizes:
            num_patches = (img_size // patch_size) ** 2
            self.num_patches[str(patch_size)] = num_patches
            
            # 基本パッチ埋め込み
            self.patch_embeds[str(patch_size)] = nn.Conv2d(
                in_chans, embed_dim // len(patch_sizes), 
                kernel_size=patch_size, stride=patch_size
            )
            
            # KA表現層追加
            self.ka_layers[str(patch_size)] = NonCommutativeKALayer(
                input_dim=embed_dim // len(patch_sizes),
                output_dim=embed_dim // len(patch_sizes),
                num_ka_terms=6
            )
        
        # 統合プロジェクション（KA拡張）
        self.fusion_ka = NonCommutativeKALayer(
            input_dim=embed_dim,
            output_dim=embed_dim,
            num_ka_terms=8
        )
        
        # ポジショナルエンコーディング
        max_patches = max(self.num_patches.values())
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, embed_dim))
        
    def forward(self, x):
        B = x.shape[0]
        embeddings = []
        
        for patch_size in self.patch_sizes:
            # パッチ埋め込み
            patch_embed = self.patch_embeds[str(patch_size)](x)
            patch_embed = patch_embed.flatten(2).transpose(1, 2)
            
            # KA表現適用
            ka_enhanced = self.ka_layers[str(patch_size)](patch_embed)
            embeddings.append(ka_enhanced)
        
        # 解像度統合
        max_len = max(emb.shape[1] for emb in embeddings)
        padded_embeddings = []
        
        for emb in embeddings:
            if emb.shape[1] < max_len:
                padding = torch.zeros(B, max_len - emb.shape[1], emb.shape[2], 
                                    device=emb.device, dtype=emb.dtype)
                emb = torch.cat([emb, padding], dim=1)
            padded_embeddings.append(emb)
        
        # 結合 + KA統合
        x = torch.cat(padded_embeddings, dim=-1)
        x = self.fusion_ka(x)
        
        # ポジショナルエンコーディング
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        return x

class NKATKAAttention(nn.Module):
    """KA表現拡張NKAT Attention"""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, temperature=0.72):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = temperature
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # 基本QKV変換
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        # KAT parameters（従来）
        self.alpha = nn.Parameter(torch.ones(num_heads))
        self.beta = nn.Parameter(torch.ones(num_heads))
        self.gamma = nn.Parameter(torch.ones(num_heads))
        
        # KA表現による非線形注意重み計算
        self.ka_attention = NonCommutativeKALayer(
            input_dim=head_dim,
            output_dim=head_dim,
            num_ka_terms=4,
            temperature=temperature
        )
        
        # 非可換アテンション融合
        self.attention_mixer = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 従来のtemperature-scaled attention
        attn_classic = (q @ k.transpose(-2, -1)) * (self.scale / self.temperature)
        
        # KAT enhancement（従来）
        attn_classic = attn_classic * self.alpha.view(1, self.num_heads, 1, 1)
        attn_classic = attn_classic + self.beta.view(1, self.num_heads, 1, 1)
        attn_classic = F.softmax(attn_classic, dim=-1)
        attn_classic = attn_classic * self.gamma.view(1, self.num_heads, 1, 1)
        
        # KA表現による非線形注意重み
        # Q, Kを統合してKA処理
        qk_combined = torch.cat([q.mean(dim=-2), k.mean(dim=-2)], dim=-1)  # B, num_heads, head_dim*2
        ka_weights = self.ka_attention(qk_combined.view(B*self.num_heads, -1))  # B*num_heads, head_dim
        ka_weights = ka_weights.view(B, self.num_heads, -1).unsqueeze(-2)  # B, num_heads, 1, head_dim
        
        # 非可換注意重み混合
        enhanced_v = v + ka_weights * v
        
        # 従来アテンション適用
        x = (attn_classic @ enhanced_v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x

class NKATKABlock(nn.Module):
    """KA表現統合NKATブロック"""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., temperature=0.72):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = NKATKAAttention(dim, num_heads=num_heads, temperature=temperature)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP with KA enhancement
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            NonCommutativeKALayer(mlp_hidden_dim, mlp_hidden_dim, num_ka_terms=6),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MultiResNKATKAModel(nn.Module):
    """KA表現統合多解像度NKATモデル"""
    
    def __init__(self, img_size=32, patch_sizes=(2, 4), num_classes=10, 
                 embed_dim=448, depth=6, num_heads=8, temperature=0.72):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Multi-resolution KA patch embedding
        self.patch_embed = MultiResKAPatchEmbedding(
            img_size=img_size, patch_sizes=patch_sizes, 
            in_chans=3, embed_dim=embed_dim
        )
        
        # KA-enhanced transformer blocks
        self.blocks = nn.ModuleList([
            NKATKABlock(embed_dim, num_heads, temperature=temperature)
            for _ in range(depth)
        ])
        
        # Classification head with KA final processing
        self.norm = nn.LayerNorm(embed_dim)
        self.ka_classifier = NonCommutativeKALayer(
            input_dim=embed_dim,
            output_dim=embed_dim // 2,
            num_ka_terms=4
        )
        self.head = nn.Linear(embed_dim // 2, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        x = self.patch_embed(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        
        # KA classification processing
        x = self.ka_classifier(x)
        x = self.head(x)
        
        return x

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EMAModel:
    """Exponential Moving Average for model parameters"""
    
    def __init__(self, model, decay=0.9998):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class NKATKATrainer:
    """KA表現統合NKAT CIFAR-10トレーナー"""
    
    def __init__(self, recovery_system=None, config=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.recovery_system = recovery_system
        self.config = config or self._default_config()
        
        # データローダー設定
        self._setup_data()
        
        # モデル設定
        self._setup_model()
        
        # 損失関数・オプティマイザー
        self._setup_training()
        
        # EMA設定
        if self.config['use_ema']:
            self.ema = EMAModel(self.model, decay=self.config['ema_decay'])
        else:
            self.ema = None
            
        # パラメータ数計算
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"🧠 NKAT-KA Model initialized")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   KA representation terms: {self.config['ka_terms']}")
        
    def _default_config(self):
        return {
            'embed_dim': 448,
            'depth': 6,
            'num_heads': 8,
            'patch_sizes': (2, 4),
            'temperature': 0.72,
            'ka_terms': 6,
            'mixup_alpha': 0.10,
            'cutmix_alpha': 0.15,
            'focal_gamma': 2.0,
            'use_ema': True,
            'ema_decay': 0.9998,
            'lr': 3e-4,
            'weight_decay': 0.05,
            'warmup_epochs': 5,
            'batch_size': 128
        }
    
    def _setup_data(self):
        # データ拡張
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # データセット
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=test_transform
        )
        
        # データローダー
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config['batch_size'], 
            shuffle=True, num_workers=4, pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.config['batch_size'], 
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        self.num_classes = 10
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
    
    def _setup_model(self):
        self.model = MultiResNKATKAModel(
            img_size=32,
            patch_sizes=self.config['patch_sizes'],
            num_classes=self.num_classes,
            embed_dim=self.config['embed_dim'],
            depth=self.config['depth'],
            num_heads=self.config['num_heads'],
            temperature=self.config['temperature']
        ).to(self.device)
    
    def _setup_training(self):
        # Focal Loss
        self.criterion = FocalLoss(gamma=self.config['focal_gamma'])
        
        # AdamW オプティマイザー
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        
        # Cosine Annealing スケジューラー
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=30, eta_min=1e-6
        )
    
    def mixup_cutmix(self, data, target):
        """Mixup and CutMix augmentation"""
        batch_size = data.size(0)
        
        if np.random.rand() < 0.5:  # Mixup
            alpha = self.config['mixup_alpha']
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
                rand_index = torch.randperm(batch_size).cuda()
                target_a, target_b = target, target[rand_index]
                mixed_data = lam * data + (1 - lam) * data[rand_index]
                return mixed_data, target_a, target_b, lam
        else:  # CutMix
            alpha = self.config['cutmix_alpha']
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
                rand_index = torch.randperm(batch_size).cuda()
                target_a, target_b = target, target[rand_index]
                
                # CutMix coordinates
                bbx1, bby1, bbx2, bby2 = self._rand_bbox(data.size(), lam)
                data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
                return data, target_a, target_b, lam
        
        return data, target, target, 1.0
    
    def _rand_bbox(self, size, lam):
        """Generate random bounding box for CutMix"""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2
    
    def train_epoch(self, epoch):
        """1エポックのトレーニング"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Training Epoch {epoch} (KA-Enhanced)')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Mixup/CutMix
            data, target_a, target_b, lam = self.mixup_cutmix(data, target)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Mixed loss
            loss = lam * self.criterion(output, target_a) + (1 - lam) * self.criterion(output, target_b)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # EMA update
            if self.ema is not None:
                self.ema.update(self.model)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Progress bar update
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'KA': '✓'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """バリデーション"""
        # EMA適用
        if self.ema is not None:
            self.ema.apply_shadow()
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Validation (KA-Enhanced)')
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # クラス別精度
                c = (predicted == target).squeeze()
                for i in range(target.size(0)):
                    label = target[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        # EMA復元
        if self.ema is not None:
            self.ema.restore()
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        # クラス別精度計算
        class_accuracies = {}
        for i in range(self.num_classes):
            if class_total[i] > 0:
                class_acc = 100. * class_correct[i] / class_total[i]
                class_accuracies[self.class_names[i]] = class_acc
        
        return avg_loss, accuracy, class_accuracies
    
    def train(self, epochs=30):
        """メイントレーニングループ"""
        print(f"🚀 Starting NKAT-KA training for {epochs} epochs")
        print(f"   Config: embed_dim={self.config['embed_dim']}, depth={self.config['depth']}")
        print(f"   Patch sizes: {self.config['patch_sizes']}")
        print(f"   KA representation: Non-commutative Kolmogorov-Arnold")
        print(f"   Temperature: {self.config['temperature']}")
        
        best_acc = 0
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        
        for epoch in range(1, epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{epochs} - KA Enhanced NKAT")
            print(f"{'='*60}")
            
            # トレーニング
            train_loss, train_acc = self.train_epoch(epoch)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # バリデーション
            val_loss, val_acc, class_accs = self.validate()
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # スケジューラー更新
            self.scheduler.step()
            
            print(f"Epoch {epoch} Results:")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            
            # 改善時にモデル保存
            if val_acc > best_acc:
                best_acc = val_acc
                if self.recovery_system:
                    checkpoint_data = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'accuracy': val_acc,
                        'loss': val_loss,
                        'config': self.config
                    }
                    
                    # EMA状態も保存
                    if self.ema is not None:
                        checkpoint_data['ema_shadow'] = self.ema.shadow
                    
                    self.recovery_system.save_checkpoint(
                        checkpoint_data, 
                        f"nkat_ka_best_epoch{epoch:03d}",
                        is_best=True
                    )
                print(f"💾 Best KA-model saved: accuracy={val_acc:.2f}%")
        
        print(f"\n🎉 NKAT-KA Training completed!")
        print(f"   Best validation accuracy: {best_acc:.2f}%")
        
        # クラス別精度詳細表示
        if class_accs:
            print(f"\n📋 Final per-class accuracy (KA-Enhanced):")
            for class_name, acc in class_accs.items():
                print(f"   {class_name:12s}: {acc:.2f}%")
        
        return {
            'best_accuracy': best_acc,
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'class_accuracies': class_accs
        }

def main():
    """メイン実行関数"""
    print("🔋 NKAT Non-Commutative Kolmogorov-Arnold CIFAR-10 System")
    print("="*65)
    print("📊 Mathematical Foundation:")
    print("   f(x) = Σᵢ φᵢ(x) ⊗ ψᵢ(Aᵢx)")
    print("   Non-commutative KA representation with multi-resolution patches")
    print("="*65)
    
    # CUDA確認
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 Using device: cuda")
        print(f"   GPU: {device_name}")
        print(f"   Memory: {memory_gb:.1f}GB")
    else:
        print("🎮 Using device: cpu")
    
    # 電源断リカバリーシステム初期化
    recovery_system = None
    if NKATPowerRecoverySystem:
        try:
            recovery_system = NKATPowerRecoverySystem(
                checkpoint_dir="checkpoints_nkat_ka_cifar10",
                auto_save_interval=300
            )
            print("🔋 Power Recovery System initialized")
        except Exception as e:
            print(f"⚠️ Recovery system initialization failed: {e}")
    
    # 設定
    config = {
        'embed_dim': 448,
        'depth': 6,
        'num_heads': 8,
        'patch_sizes': (2, 4),
        'temperature': 0.72,
        'ka_terms': 6,
        'mixup_alpha': 0.10,
        'cutmix_alpha': 0.15,
        'focal_gamma': 2.0,
        'use_ema': True,
        'ema_decay': 0.9998,
        'lr': 3e-4,
        'weight_decay': 0.05,
        'warmup_epochs': 5,
        'batch_size': 128
    }
    
    # トレーナー初期化
    trainer = NKATKATrainer(recovery_system=recovery_system, config=config)
    
    # トレーニング実行
    try:
        results = trainer.train(epochs=25)
        
        # 結果表示
        print(f"\n🎯 Final KA-Enhanced Results:")
        print(f"   Best Accuracy: {results['best_accuracy']:.2f}%")
        
        if results['best_accuracy'] >= 80.0:
            print("🎉 KA TARGET ACHIEVED! Accuracy ≥ 80%")
            print("🧮 Non-commutative Kolmogorov-Arnold representation successful!")
        elif results['best_accuracy'] >= 78.0:
            print("✅ Multi-resolution target achieved! Accuracy ≥ 78%")
        else:
            print(f"📈 Progress: {results['best_accuracy']:.2f}% (Target: 78%)")
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        if recovery_system:
            recovery_system.emergency_save({
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'message': 'Interrupted by user'
            })
    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        if recovery_system:
            recovery_system.emergency_save({
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'error': str(e)
            })
        raise
    
    finally:
        # GPU メモリクリーンアップ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 CUDA memory cleaned up")

if __name__ == "__main__":
    main() 