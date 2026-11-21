@echo off
chcp 936 >nul
cd /d "%~dp0"
title T1 wan zheng shi yan
echo ============================================
echo   T1 共享单车项目   完整实验
echo   （约 6 分钟，论文里的数字都来自这一次）
echo ============================================
echo.
echo 注意：会覆盖 results / figures / tables 里的旧文件。
echo 想保留旧结果的话，先复制一份再继续。
echo.
pause

python -c "import numpy,pandas,sklearn,matplotlib" >nul 2>nul
if errorlevel 1 (
    echo 正在安装依赖 ...
    python -m pip install -r requirements.txt
)

echo.
echo 开始运行（约 6 分钟，窗口会一直刷文字，属正常）...
echo --------------------------------------------
python run_all.py
if errorlevel 1 goto FAIL

echo --------------------------------------------
echo [OK] 全部完成。产物位置：
echo      figures\   6 张图（PDF + PNG）
echo      tables\    tab1-tab4
echo      results\   E1-E7 原始数字 + run_manifest.json
echo.
echo 请保留 results\run_manifest.json，它是这次实验的记录。
if exist "results" start "" "results"
pause
exit /b 0

:FAIL
echo --------------------------------------------
echo [X] 运行失败，请截图发群。
pause
exit /b 1
