::encoding: gbk
@echo off
::chcp 65001
color 0F
echo ————请先使用pull.bat合并————
set /p input="是否拉取检查合并？（y：检查，n：不检查）"
if /i %input% == y call pull.bat
:pull
git pull
set error=%errorlevel%
if %error% neq 0 echo 错误代码：%error%
if %error% neq 0 echo.
if %error% neq 0 goto pull
echo ————拉取完成！合并中……————
:push
git push
set error=%errorlevel%
if %error% neq 0 echo 错误代码：%error%
if %error% neq 0 echo.
if %error% neq 0 goto push
echo ————拉取完成！按任意键退出————
pause