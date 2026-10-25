@echo off
rem 本地打包：从 ncnn-spike 构建目录取最新 ncnn_shim.dll -> dist/win-x64 -> dotnet pack
setlocal
set SPIKE_DIR=D:\bgiprro\ncnn-spike
set DIST=%~dp0dist\win-x64
set ART=%~dp0artifacts

if not exist "%SPIKE_DIR%\ncnn_shim.dll" (
  echo [错误] 未找到 %SPIKE_DIR%\ncnn_shim.dll，请先运行 build_shim.bat
  exit /b 1
)

copy /y "%SPIKE_DIR%\ncnn_shim.dll" "%DIST%\ncnn_shim.dll" >nul
dotnet pack "%~dp0pack\Guamian.BetterGI.NcnnShim.csproj" -c Release -o "%ART%" --nologo
if errorlevel 1 exit /b 1
echo [完成] 原生包已输出到 %ART%
