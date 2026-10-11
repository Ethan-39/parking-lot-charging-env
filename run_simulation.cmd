@echo off
setlocal
chcp 65001 >nul
set "charging_base_python=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
set "charging_venv=%~dp0..\work\desktop_venv"
set "charging_python=%charging_venv%\Scripts\python.exe"
if not exist "%charging_python%" (
    echo 首次启动：正在创建本程序的 Python 环境，请稍候。
    "%charging_base_python%" -I -X utf8 -m venv "%charging_venv%"
    if errorlevel 1 goto setup_failed
)

"%charging_python%" -I -X utf8 -c "from gymnasium import spaces; import numpy" >nul 2>&1
if errorlevel 1 (
    echo 正在安装所需组件，首次安装需要联网，请稍候。
    "%charging_python%" -I -X utf8 -m pip install --disable-pip-version-check -r "%~dp0requirements.txt"
    if errorlevel 1 goto setup_failed
)

echo 开始运行停车场充电模拟。
"%charging_python%" -I -X utf8 "%~dp0parking_lot_charging_env.py" %*
set "charging_exit=%errorlevel%"
if "%charging_exit%"=="0" (
    echo 模拟已完成，结果见上方。
) else (
    echo 程序运行失败，请查看上方的报错信息。
)
goto finished

:setup_failed
set "charging_exit=1"
echo Python 环境准备失败，请查看上方的报错信息。

:finished
if not "%parking_no_pause%"=="1" (
    echo 按任意键关闭此窗口。
    pause >nul
)
exit /b %charging_exit%
