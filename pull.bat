::encoding: utf8
@echo off
chcp 65001
color 0F
:pull
git pull
set error=%errorlevel%
if %error% neq 0 echo 错误代码：%error%
if %error% neq 0 echo.
if %error% neq 0 goto pull
echo ————拉取完成，请检查！————
pause