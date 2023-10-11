::encoding: gbk
@echo off
color F0
:pull
git pull
if %errorlevel% neq 0 goto pull
:push
echo �������ȡ��
pause