@echo off
rem 本地打包：build_shim.bat 的产物（dist/win-x64/ncnn_shim.dll）-> dotnet pack
setlocal
set DIST=%~dp0dist\win-x64
set ART=%~dp0artifacts

if not exist "%DIST%\ncnn_shim.dll" (
  echo [错误] 未找到 %DIST%\ncnn_shim.dll，请先运行 build_shim.bat
  exit /b 1
)

dotnet pack "%~dp0pack\Guamian.BetterGI.NcnnShim.csproj" -c Release -o "%ART%" --nologo
if errorlevel 1 exit /b 1
echo [完成] 原生包已输出到 %ART%
