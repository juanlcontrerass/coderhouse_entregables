@echo off
rem Lanzador para Windows: ejecuta run.ps1 sin depender de la política de ejecución de PowerShell.
rem   run.cmd            levanta el sistema
rem   run.cmd down       detiene todo
rem   run.cmd logs       sigue los logs de la API y el worker
rem   run.cmd status     estado de los contenedores
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
exit /b %ERRORLEVEL%
