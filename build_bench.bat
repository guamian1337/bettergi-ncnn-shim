@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d D:\bgiprro\ncnn-spike
cl /nologo /utf-8 /O2 /MD /EHsc /I ncnn-20261020-windows-vs2022\x64\include /I ncnn-20261020-windows-vs2022\x64\include\ncnn bench.cpp /link /LIBPATH:ncnn-20261020-windows-vs2022\x64\lib /LIBPATH:. ncnn.lib glslang.lib SPIRV.lib OSDependent.lib MachineIndependent.lib GenericCodeGen.lib glslang-default-resource-limits.lib vulkan-1.lib /OUT:ocr_rec_bench.exe
