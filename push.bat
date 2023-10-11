@echo off
color F0
:pull
git pull
if %errorlevel% neq 0 goto pull
:push
echo 已完成拉取！
git push
if %errorlevel% neq 0 goto push
echo 已完成！
pause