::encoding: gbk
@echo off
::chcp 65001
color 0F
echo ������������ʹ��pull.bat�ϲ���������
set /p input="�Ƿ���ȡ���ϲ�����y����飬n������飩"
if /i %input% == y call pull.bat
:pull
git pull
set error=%errorlevel%
if %error% neq 0 echo ������룺%error%
if %error% neq 0 echo.
if %error% neq 0 goto pull
echo ����������ȡ��ɣ��ϲ��С�����������
:push
git push
set error=%errorlevel%
if %error% neq 0 echo ������룺%error%
if %error% neq 0 echo.
if %error% neq 0 goto push
echo ����������ȡ��ɣ���������˳���������
pause