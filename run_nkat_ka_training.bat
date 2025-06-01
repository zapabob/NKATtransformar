@echo off
echo 🔋 NKAT Non-Commutative Kolmogorov-Arnold CIFAR-10 System
echo ============================================================
echo 📊 Mathematical Foundation:
echo    f(x) = Σᵢ φᵢ(x) ⊗ ψᵢ(Aᵢx)
echo    Non-commutative KA representation with multi-resolution patches
echo ============================================================
echo.

:: Python環境確認
echo 🐍 Checking Python environment...
py -3 --version
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

:: CUDA確認
echo ⚡ Checking CUDA environment...
py -3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo 🚀 Starting NKAT-KA Enhanced Training...
echo    Target: CIFAR-10 accuracy ≥ 80% (with KA representation)
echo    Features: Non-commutative Kolmogorov-Arnold theory
echo              Multi-resolution patches (2×2 & 4×4)
echo              Focal Loss, EMA, Power Recovery
echo              Mathematical rigor + Deep Learning
echo.

:: トレーニング実行
py -3 nkat_noncommutative_kolmogorov_arnold.py

echo.
echo 🎯 NKAT-KA Training completed!
echo 🧮 Check if non-commutative KA representation enhanced performance.
echo.
pause 