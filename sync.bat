::encoding: gbk
@echo off
::chcp 65001
color 0F
echo ��������������������ʹ��pull.bat�ϲ�����������������
set /p input="�Ƿ���ȡ���ϲ�����y����飬n������飩"
if /i %input% == y call pull.bat
:pull
git pull
if %errorlevel% neq 0 (
    echo ������룺%errorlevel%
    echo.
    goto pull)
echo ������������������ȡ��ɣ��ϲ��С�������������������
:push
git push
if %errorlevel% neq 0 (
    echo ������룺%errorlevel%
    echo.
    goto push)
echo ������������������ȡ��ɣ���������˳�����������������
pause >nul