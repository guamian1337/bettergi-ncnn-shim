@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d D:\bgiprro\ncnn-spike
rem /Zi + /DEBUG:PDB 生成 ncnn_shim.pdb：与 DLL 一起部署（放 DLL 同目录），
rem 供 VTune / WinDbg 符号化 shim 与静态链入的 glslang/ncnn 内部函数，排查编译与上传热点。
cl /nologo /utf-8 /O2 /MD /LD /EHsc /Zi /I ncnn-20261020-windows-vs2022\x64\include /I ncnn-20261020-windows-vs2022\x64\include\ncnn ncnn_shim.cpp /link /DEBUG /LIBPATH:ncnn-20261020-windows-vs2022\x64\lib /LIBPATH:. ncnn.lib glslang.lib SPIRV.lib OSDependent.lib MachineIndependent.lib GenericCodeGen.lib glslang-default-resource-limits.lib vulkan-1.lib /OUT:ncnn_shim.dll
