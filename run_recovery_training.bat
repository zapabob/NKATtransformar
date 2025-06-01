@echo off
echo.
echo ==========================================
echo   NKAT 電源断リカバリーシステム 起動
echo ==========================================
echo.

REM CUDA環境確認
echo 🎮 CUDA環境確認中...
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
if %ERRORLEVEL% neq 0 (
    echo ❌ CUDA環境が利用できません
    pause
    exit /b 1
)

echo.
echo ✅ CUDA環境確認完了
echo.

REM Pythonバージョン確認
echo 🐍 Python環境確認中...
py -3 --version
if %ERRORLEVEL% neq 0 (
    echo ❌ Python 3が利用できません
    pause
    exit /b 1
)

echo.
echo ✅ Python環境確認完了
echo.

REM リカバリーシステムテスト
echo 🔋 リカバリーシステムテスト実行中...
py -3 test_recovery_system.py
if %ERRORLEVEL% neq 0 (
    echo ❌ リカバリーシステムテストに失敗しました
    pause
    exit /b 1
)

echo.
echo ✅ リカバリーシステムテスト完了
echo.

REM エポック数入力
set /p epochs="トレーニングエポック数を入力してください (デフォルト: 50): "
if "%epochs%"=="" set epochs=50

echo.
echo 🚀 CIFAR-10 トレーニング開始...
echo    エポック数: %epochs%
echo    リカバリーシステム: 有効
echo.

REM CIFAR-10トレーニング実行
py -3 nkat_cifar10_recovery_training.py %epochs%

echo.
echo ==========================================
echo   トレーニング完了
echo ==========================================
echo.

pause 