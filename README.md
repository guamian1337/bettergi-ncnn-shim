# bettergi-ncnn-shim

BetterGI 的 ncnn Vulkan/CPU 推理胶水层（`ncnn_shim.dll`）：纯 C ABI 导出，静态链接 ncnn + glslang + SPIRV。
C# 侧互操作契约见主仓库 `BetterGenshinImpact/Core/Recognition/Ncnn/INcnnShim.cs`。

## 仓库结构

| 文件 | 用途 |
|---|---|
| `ncnn_shim.cpp` | 胶水层源码（唯一需要编译的产物） |
| `build_shim.bat` | 本地构建 `ncnn_shim.dll` |
| `build_yolo.bat` / `build_bench.bat` | 辅助工具（形状验证 / C++ 基准），可选 |
| `convert_model.py` | ONNX → ncnn 模型转换与对拍脚本 |
| `pack/` + `pack_local.bat` | NuGet 原生包打包 |
| `.github/workflows/publish.yml` | tag 触发的 CI 发布（Trusted Publishing） |

## 本地构建

1. 下载 [ncnn releases](https://github.com/Tencent/ncnn/releases) 的 windows-vs2022 x64 完整包（含 Vulkan），
   解压到 `D:\bgiprro\ncnn-spike\`（与 build 脚本内路径一致，版本需与
   `.github/workflows/publish.yml` 的 `NCNN_VERSION` 相同）；
2. 把 `ncnn_shim.cpp` 复制到该目录，运行 `build_shim.bat` 得到 `ncnn_shim.dll`。

> 注意：vulkan-1.lib 不在 Windows SDK 内，脚本用 dumpbin 从 `System32\vulkan-1.dll` 生成导入库。

## 打包与发布

### 本地快速迭代（不发布）

```bat
pack_local.bat
```

从 `ncnn-spike` 构建目录取最新 dll → `dist/win-x64/` → `dotnet pack` → `artifacts/`。
主仓库 `nuget.config` 已配置 `..\bettergi-ncnn-shim\artifacts` 为目录源，
主工程构建时自动把 dll 部署到 `bin\...\Assets\Model\PaddleOCR\Ncnn\`。

### 正式发布（Trusted Publishing，无 API Key）

1. nuget.org 包页 → Trusted Publishing 政策：Owner（nuget.org 用户名）、
   GitHub 仓库、Workflow = `publish.yml`、Environment = `nuget-org`；
2. 提升版本号（`pack/Guamian.BetterGI.NcnnShim.csproj` 的 `<Version>`）；
3. `git tag v0.1.1 && git push origin v0.1.1` —— CI 自动构建、打包、推送。

> 版本一经发布不可覆盖，每次发布必须递增。

## 主仓库集成

`BetterGenshinImpact.csproj` 引用 `Guamian.BetterGI.NcnnShim`；
还原后包内 `buildTransitive/Guamian.BetterGI.NcnnShim.targets` 在构建时把 dll
部署到 `$(OutDir)Assets\Model\PaddleOCR\Ncnn\`（即 `NcnnAvailability.ShimRelativePath` 的加载位置）。
运行库缺失时 ncnn 后端抛 `NcnnUnavailableException`，不影响 ONNX 路径。

## 模型转换（convert_model.py）

```bat
python convert_model.py <模型>\xxx.onnx                 :: 修复 → 转换 → 对拍 → 部署
python convert_model.py <模型>\xxx.onnx --skip-deploy   :: 只验证不部署
```

流程：元数据检查 → 拓扑排序 → fp16→fp32 → checker → pnnx → 层型检查 → 数值对拍（vs ORT）→ 部署。
依赖 `onnx onnxruntime numpy` 与本机 `pnnx.exe`、`yolo_check.exe`（`--pnnx`/`--yolo-check` 可覆盖路径）。

> OCR 动态模型（det_dyn/rec_dyn）用 pnnx **双具体形状**转换：
> `inputshape=[1,3,48,320] inputshape2=[1,3,48,960]`（rec 宽度动态同理），不要写 `-1` 维度。
> det 虽然全动态转换，但生产固定**正方形**画布——ncnn Vulkan 对非方形动态输入存在引擎缺陷
> （非方形输出错乱，CPU 正确），详见主仓库 `NcnnDet`/`NcnnRec`。
> 字典与 ONNX 路线必须同源，逐字一致。

## C API 概览

完整导出表见 `ncnn_shim.cpp` 与 `INcnnShim.cs`（二者一一对应）。要点：

- `shim_create_ex(ShimNetOptions*)`：版本化选项（首字段 `size` 由调用方填 `sizeof`），
  参数必须在 `load_param` 之前写入 `net->opt`（fp16/int8/winograd/packing 加载后设置无效）；
- `shim_infer_ex(handle, inputs, n, outputNames, m, extractor, tensors)`：多输入多输出，
  输入支持 float CHW 与常见像素格式，输出按 `w*h*c` 紧凑拷贝（`count` 不含对齐填充）；
- 设备枚举：`shim_get_device_count` / `shim_get_device_name` / `shim_get_device_info` /
  `shim_get_heap_budget`（首次调用初始化 Vulkan 实例，随进程常驻）。

### 硬性约束（违反即崩溃或静默无效）

- **`use_bf16_packed` 保持 `false`**（ncnn 默认值）：部分设备 + cooperative matrix 下打开会
  着色器编译失败并直接崩溃进程；
- 输入 blob 名以 `shim_get_input_name` 为准，不要硬编码；
- `gpuIndex < 0` 表示纯 CPU 推理（`numThreads` 生效），`>= 0` 为 Vulkan 设备索引。
