::encoding: utf8
@echo off
@chcp 65001
@color 0F
:pull
git pull
echo.
if %errorlevel% neq 0 goto pull
:push
echo ————拉取完成！请检查合并冲突后按任意键合并！————
@pause
git push
echo.
if %errorlevel% neq 0 goto push
echo ————拉取完成！按任意键退出————
@pause