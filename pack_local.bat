@echo off
rem Pack build_shim.bat output (dist\win-x64\ncnn_shim.dll) into a NuGet package.
setlocal
set DIST=%~dp0dist\win-x64
set ART=%~dp0artifacts

if not exist "%DIST%\ncnn_shim.dll" (
  echo [ERROR] %DIST%\ncnn_shim.dll not found. Run build_shim.bat first.
  exit /b 1
)

dotnet pack "%~dp0pack\Guamian.BetterGI.NcnnShim.csproj" -c Release -o "%ART%" --nologo
if errorlevel 1 exit /b 1
echo [OK] package written to %ART%
