@echo off
setlocal
rem ============================================================
rem  ncnn_shim.dll local build
rem  Prerequisites:
rem    1. Visual Studio 2022+ (C++ workload; located via vswhere)
rem    2. LunarG Vulkan SDK (https://vulkan.lunarg.com/sdk/home)
rem       provides official vulkan-1.lib; VULKAN_SDK env set by installer.
rem       If env is stale (SDK installed after terminal opened), the script
rem       auto-detects the newest version under C:\VulkanSDK.
rem    3. ncnn prebuilt libs (Tencent/ncnn releases, windows-vs2022 x64 full)
rem       Usage: build_shim.bat [ncnn_dir], default D:\bgiprro\ncnn-spike
rem  Output: dist\win-x64\ncnn_shim.dll (+ .pdb for VTune/WinDbg)
rem ============================================================

set SHIM_DIR=%~dp0
set NCNN_DIR=%~1
if "%NCNN_DIR%"=="" set NCNN_DIR=D:\bgiprro\ncnn-spike

rem ---- Vulkan SDK ----
if "%VULKAN_SDK%"=="" (
  for /f "delims=" %%d in ('dir /b /ad /o-n C:\VulkanSDK 2^>nul') do (
    if not defined VULKAN_SDK if exist "C:\VulkanSDK\%%d\Lib\vulkan-1.lib" set "VULKAN_SDK=C:\VulkanSDK\%%d"
  )
)
if "%VULKAN_SDK%"=="" (
  echo [ERROR] Vulkan SDK not found.
  echo         Install LunarG Vulkan SDK: https://vulkan.lunarg.com/sdk/home
  echo         Then reopen the terminal, or ensure C:\VulkanSDK\^<ver^>\Lib\vulkan-1.lib exists.
  exit /b 1
)
if not exist "%VULKAN_SDK%\Lib\vulkan-1.lib" (
  echo [ERROR] %VULKAN_SDK%\Lib\vulkan-1.lib not found. Install the full LunarG Vulkan SDK.
  exit /b 1
)

rem ---- Visual Studio via vswhere ----
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo [ERROR] vswhere not found. Install Visual Studio 2022+ with C++ workload.
  exit /b 1
)
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_PATH=%%i"
if "%VS_PATH%"=="" (
  echo [ERROR] No Visual Studio instance with C++ toolchain found.
  exit /b 1
)
call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

rem ---- ncnn prebuilt package ----
set "NCNN_PKG="
for /d %%d in ("%NCNN_DIR%\ncnn-*-windows-vs2022") do set "NCNN_PKG=%%d"
if "%NCNN_PKG%"=="" (
  echo [ERROR] No ncnn-*-windows-vs2022 directory under %NCNN_DIR%.
  echo         Download the x64 full package from https://github.com/Tencent/ncnn/releases
  exit /b 1
)
if not exist "%NCNN_PKG%\x64\lib\ncnn.lib" (
  echo [ERROR] %NCNN_PKG%\x64\lib\ncnn.lib not found. Download the full Vulkan-enabled package.
  exit /b 1
)

if not exist "%SHIM_DIR%dist\win-x64" mkdir "%SHIM_DIR%dist\win-x64"

cd /d "%NCNN_DIR%"
rem /Zi + /DEBUG:PDB: deploy .pdb next to the dll for VTune/WinDbg symbolication
cl /nologo /utf-8 /O2 /MD /LD /EHsc /Zi ^
  /I "%NCNN_PKG%\x64\include" ^
  /I "%NCNN_PKG%\x64\include\ncnn" ^
  "%SHIM_DIR%ncnn_shim.cpp" ^
  /link /DEBUG ^
  /LIBPATH:"%NCNN_PKG%\x64\lib" ^
  /LIBPATH:"%VULKAN_SDK%\Lib" ^
  ncnn.lib glslang.lib SPIRV.lib OSDependent.lib MachineIndependent.lib GenericCodeGen.lib ^
  glslang-default-resource-limits.lib vulkan-1.lib ^
  /OUT:"%SHIM_DIR%dist\win-x64\ncnn_shim.dll"
if errorlevel 1 exit /b 1
echo [OK] %SHIM_DIR%dist\win-x64\ncnn_shim.dll
