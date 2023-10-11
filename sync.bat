::encoding: utf8
@echo off
@chcp 65001
@color 0F
echo ————请先使用pull.bat合并————
@pause
:pull
git pull
set error = %errorlevel%
echo %error%
if %error% neq 0 goto pull
echo ————拉取完成！合并中……————
:push
git push
set error = %errorlevel%
echo %error%
if %error% neq 0 goto push
echo ————拉取完成！按任意键退出————
@pause