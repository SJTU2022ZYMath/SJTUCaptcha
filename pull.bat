::encoding: utf8
@echo off
chcp 65001
color 0F
:pull
git pull
if %errorlevel% neq 0 (
    echo 错误代码：%errorlevel%
    echo.
    goto pull)
echo ————————拉取完成，请检查！————————
pause >nul