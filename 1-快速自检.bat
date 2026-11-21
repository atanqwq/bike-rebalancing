@echo off
chcp 936 >nul
cd /d "%~dp0"
title T1 kuaishu zhi jian
echo ============================================
echo   T1 共享单车项目   快速自检
echo   （约 30 秒，只验证环境能不能跑通）
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 goto NOPY

for /f "tokens=2" %%v in ('python --version 2^>^&1') do echo [1/4] Python 版本 %%v  OK

echo [2/4] 检查依赖库 ...
python -c "import numpy,pandas,sklearn,matplotlib" >nul 2>nul
if errorlevel 1 goto INSTALL
echo       依赖齐全，跳过安装。
goto CHECKCFG

:INSTALL
echo       缺少依赖，正在安装（第一次 1-3 分钟）...
python -m pip install -r requirements.txt
if errorlevel 1 goto PIPFAIL

:CHECKCFG
echo [3/4] 检查配置文件 ...
python config.py
if errorlevel 1 goto RUNFAIL

echo.
echo [4/4] 跑全流程（约 30 秒）...
echo --------------------------------------------
python run_all.py --fast
if errorlevel 1 goto RUNFAIL

echo --------------------------------------------
echo [OK] 成功！
echo      请打开 figures 文件夹，应该看到 6 张图：
echo      fig1 fig2 fig3 fig4 fig5 fig6
echo.
if exist "figures" start "" "figures"
echo 遇到问题请把本窗口截图发群里。
pause
exit /b 0

:NOPY
echo [X] 没有找到 python。
echo     请先安装 Python：https://www.python.org/downloads/
echo     安装时务必勾选 "Add python.exe to PATH"，装完关掉所有黑窗口重开。
pause
exit /b 1

:PIPFAIL
echo.
echo [X] 依赖安装失败。换国内镜像再试：
echo     python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pause
exit /b 1

:RUNFAIL
echo --------------------------------------------
echo [X] 运行失败。请把报错往上 20 行截图发群里。
pause
exit /b 1
