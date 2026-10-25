@echo off
setlocal
rem ============================================================
rem  ncnn_shim.dll 本地构建
rem  前置：
rem    1. Visual Studio 2022+（含 C++ 工具链，自动经 vswhere 定位）
rem    2. LunarG Vulkan SDK（https://vulkan.lunarg.com/sdk/home）
rem       安装后 VULKAN_SDK 环境变量自动设置，Lib 下含官方 vulkan-1.lib
rem    3. ncnn 预编译库（Tencent/ncnn releases，windows-vs2022 x64 完整包）
rem       用法：build_shim.bat [ncnn 库目录]，缺省 D:\bgiprro\ncnn-spike
rem         目录内需有 ncnn-<版本>-windows-vs2022\x64\{include,lib}
rem  产物：dist\win-x64\ncnn_shim.dll（+ pdb，供 VTune/WinDbg 符号化）
rem ============================================================

set SHIM_DIR=%~dp0
set NCNN_DIR=%~1
if "%NCNN_DIR%"=="" set NCNN_DIR=D:\bgiprro\ncnn-spike

rem ---- Vulkan SDK（官方 vulkan-1.lib 所在）----
if "%VULKAN_SDK%"=="" (
  echo [错误] 未检测到 VULKAN_SDK 环境变量。
  echo        请安装 LunarG Vulkan SDK：https://vulkan.lunarg.com/sdk/home
  echo        安装完成后重新打开终端，环境变量自动生效。
  exit /b 1
)
if not exist "%VULKAN_SDK%\Lib\vulkan-1.lib" (
  echo [错误] %VULKAN_SDK%\Lib\vulkan-1.lib 不存在，请安装完整版 LunarG Vulkan SDK。
  exit /b 1
)

rem ---- Visual Studio（vswhere 定位最新含 C++ 工具链的实例）----
set VSWHERE="%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist %VSWHERE% (
  echo [错误] 未找到 vswhere，请安装 Visual Studio 2022 或更新（含 C++ 桌面开发负载）。
  exit /b 1
)
for /f "usebackq tokens=*" %%i in (`%VSWHERE% -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_PATH=%%i"
if "%VS_PATH%"=="" (
  echo [错误] 未找到含 C++ 工具链的 Visual Studio 实例。
  exit /b 1
)
call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

rem ---- ncnn 预编译库 ----
for /d %%d in ("%NCNN_DIR%\ncnn-*-windows-vs2022") do set "NCNN_PKG=%%d"
if "%NCNN_PKG%"=="" (
  echo [错误] %NCNN_DIR% 下未找到 ncnn-*-windows-vs2022 预编译库目录。
  echo        从 https://github.com/Tencent/ncnn/releases 下载 x64 完整包解压至此。
  exit /b 1
)
if not exist "%NCNN_PKG%\x64\lib\ncnn.lib" (
  echo [错误] %NCNN_PKG%\x64\lib\ncnn.lib 不存在，请下载完整版（含 Vulkan）。
  exit /b 1
)

if not exist "%SHIM_DIR%dist\win-x64" mkdir "%SHIM_DIR%dist\win-x64"

cd /d "%NCNN_DIR%"
rem /Zi + /DEBUG:PDB：与 DLL 一起部署（放 DLL 同目录），供 VTune/WinDbg 符号化
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
echo [完成] %SHIM_DIR%dist\win-x64\ncnn_shim.dll
