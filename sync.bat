::encoding: gbk
@echo off
color 0F
:pull
git pull
if %errorlevel% neq 0 goto pull
:push
echo ����������ȡ��ɣ�����ϲ���ͻ��������ϲ�����������
@pause
git push
if %errorlevel% neq 0 goto push
echo ����������ȡ��ɣ���������˳���������
@pause