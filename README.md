# ncnn shim（Vulkan / CPU，实验）

PP-OCR / YOLO 的 ncnn 推理 shim：C API（ncnn_shim.dll）+ C++ 基准 + C# P/Invoke 检查。
支持两种计算后端：Vulkan 显卡加速（`gpuIndex >= 0`）与 ncnn 原生 CPU 推理（`gpuIndex < 0`，此时 `numThreads` 决定并行度）。
B580 实测：rec@480 平均 10.7ms（对比 ORT DirectML EP 同输入 ~20ms 地板，2 倍）；CPU 模式 rec@320 实测 68ms（i9 级别处理器）。

## 构建

1. 下载 ncnn windows-vs2022 release（20261020 或更新，x64 完整包，含 Vulkan）解压到本目录的
   `ncnn-20261020-windows-vs2022/`；模型放 `models/`（PP-OCRv3 det/rec ncnn，来自 FeiGeChuanShu/ncnn_paddleocr）。
2. 把本目录的 `ncnn_shim.cpp` 复制到 ncnn 包的上一级目录（`build_shim.bat` 的工作目录），
   运行 `build_shim.bat`（生成 ncnn_shim.dll）、`build_bench.bat`（生成 C++ 基准）。
3. `csharp-check/` 为 C# P/Invoke 闭环验证工程。
4. 部署：把生成的 `ncnn_shim.dll` 覆盖到运行目录 `Assets\Model\PaddleOCR\Ncnn\`
   （Debug 构建下 `bin\Debug\net8.0-windows10.0.22621.0\` 与 `bin\x64\Debug\net8.0-windows10.0.22621.0\` 各一份）。

## 导出接口

| 导出 | 说明 |
|---|---|
| `shim_create(param, bin, gpuIndex, numThreads)` | 旧接口：`gpuIndex >= 0` 使用 Vulkan 设备；`< 0` 为纯 CPU 推理，`numThreads` 生效 |
| `shim_infer` / `shim_destroy` / `shim_free_buffer` | 旧接口：单输入单输出同步推理与资源释放 |
| `shim_create_ex(const ShimNetOptions*)` | 版本化选项：首字段 `size` 由调用方填 `sizeof`，其余字段按需覆盖 `ncnn::Option`；参数在 `load_param` 前写入 |
| `shim_get_default_options(ShimNetOptions*)` | 读回当前 ncnn 的 `Option` 默认值，供 C# 侧校验内置默认是否与库一致 |
| `shim_get_input_count` / `shim_get_input_name` | 真实输入 blob 名（替代硬编码 `in0`） |
| `shim_get_output_count` / `shim_get_output_name` | 真实输出 blob 名（替代硬编码 `out0`） |
| `shim_infer_ex(handle, inputs, n, outputNames, m, extractor, tensors)` | 多输入多输出；`ShimInput` 支持 float CHW 与 RGB888/BGR888/RGBA8888/BGRA8888/NV12；`ShimTensor` 返回 dims/w/h/d/c/elempack/elembits/cstep/count |
| `shim_infer_pixels(handle, pixels, fmt, w, h, stride, input, output, out)` | 单输入像素直输便捷入口（NV12 先 `resize_bilinear_yuv420sp` 再转 RGB） |
| `shim_get_device_count()` | Vulkan 设备数量；首次调用会 `create_gpu_instance()`，实例随进程常驻 |
| `shim_get_device_name(index, buf, len)` | 设备名；返回类型码 0 独显 / 1 核显 / 2 虚拟显卡 / 3 CPU，索引无效返回 -1 |
| `shim_get_default_device()` | ncnn 默认设备索引（独显优先），无设备返回 -1 |
| `shim_get_device_info(index, ShimDeviceInfo*)` | 一次返回名称、类型、vendor/device id、API/驱动版本、驱动名、计算/传输队列、统一队列、ReBAR、image storage、shader local memory、FP16/INT8/INT16/BF16 能力、subgroup 能力、共享内存、性能评分、显存预算 |
| `shim_get_heap_budget(index)` | 仅显存预算（MB），0 表示未知 |

`shim_infer_ex` 输出统一走 `Extractor::extract(name, mat, 0)`（fp16/bf16/packing 转回 fp32 且 unpacked），
按 `w*h*c` 逐通道拷贝并用 `cstep` 跳过对齐填充，`count` 为紧凑元素数而非 `total()`。

不再支持旧版 DLL：`NcnnShim` 构造时绑定全部必需导出，缺任一导出直接抛 `NotSupportedException`；
`NcnnShim.IsSupported(dllPath)` 只探测导出表供启动校验与设置页使用。设备/详情/显存导出缺失同样直接报错，
不再降级为设备索引下拉或空详情（需要旧版兼容时请更换 DLL，而不是让代码猜测）。

## 三套调优参数（检测 / 识别 / YOLO）

设置页「推理引擎设置 → ncnn → 精度与调优」提供三套**完全独立**的参数，各自创建会话时下发：

| 分组 | 字段 |
|---|---|
| 并行 | `threads`、`parallelSessions`（1–4，>1 时同一用途并行多会话）、`openmpBlocktime` |
| 精度 | `useFp16Packed/Storage/Arithmetic`、`useInt8Inference/Packed/Storage/Arithmetic`、`useBf16Storage` |
| 卷积优化 | `useWinogradConvolution`、`useWinograd23/43/63`、`useSgemmConvolution`、`usePackingLayout` |
| Vulkan 细节 | `useCooperativeMatrix`、`useFp16Uniform`、`useInt16Packed`、`useShaderLocalMemory`、`useA53A55OptimizedKernel`、`usePixelInput` |

生效顺序：`shim_create_ex` 先按字段写入 `net->opt`，再 `load_param` / `load_model`；
fp16 / int8 / winograd / packing 若在加载后才设置会被 ncnn 忽略。

注意事项：

- INT8 相关项只有配套的 **int8 量化模型**才有收益，未量化模型开启后数值不变、速度不变。
- `usePackingLayout` 只在 ARM 等支持打包的平台上收益明显，x86 显卡通常无变化。
- `usePixelInput` 要求模型输入接受 0–255（自带归一化），默认关闭；开启后 YOLO 走像素直输，省掉一次 CHW 浮点写入。
- 调优参数改动后需「应用并重启」，因为会话在启动期创建。

### 默认值对齐与已知坑

- 内置默认值必须与 ncnn 库默认一致：`InferenceRuntimeService.PrepareStartup` 在启用 ncnn 时调用
  `shim_get_default_options` 做字段级对齐，只校正「用户没改动过」的字段（线程数属产品默认，不参与对齐）。
- **`use_bf16_packed` 必须保持 `false`**（这也是 ncnn 默认值）：B580 + ncnn 20261020 打开后，
  cooperative matrix 卷积着色器会编译失败（`cannot convert from coopmat<float16_t> to coopmat<float>`），
  紧接着进程直接崩溃。`NcnnTuningSettings.ToNetOptions` 已显式关闭，别再打开。
- 原生会话不要放进单元测试宿主跑：主程序有单实例检测，重复进程会自行退出；ncnn 全局对象在进程退出阶段
  还会触发 `FAST_FAIL_FATAL_APP_EXIT(0xc0000409)`，整轮测试连同结果一起中止。Vulkan 用例需显式设置
  环境变量 `BETTERGI_NCNN_VULKAN_TESTS=1` 单独运行（见 `NcnnTestAssets`），默认只跑 CPU 用例。

## 注意

- ncnn.lib 静态链接 glslang/SPIRV（Vulkan shader 编译）；vulkan-1.lib 不在 Windows SDK 里，
  build 脚本用 dumpbin 从 System32\vulkan-1.dll 自造导入库。
- DllImport 的 dll 路径必须是编译期常量；模型输入层名 `input`、输出层名 `out`（v3 模型）。
- 尚未生产化：v6 模型转换、det 接入、BetterGI 接口层、精度对拍均为待办。

## 一键转换脚本（convert_model.py）

把"新角色/新技能 = 重训 onnx → 换 ncnn 产物"的整条流水线脚本化：

```bat
python Profile\ncnn-shim\convert_model.py <模型>\xxx.onnx                # 修复→转换→对拍→部署
python Profile\ncnn-shim\convert_model.py <模型>\xxx.onnx --skip-deploy  # 只验证不部署
```

流程：元数据检查 → 拓扑排序 → fp16→fp32 → checker → pnnx → 层型检查（禁止 Tensor.to/裸 Conv）
→ 数值对拍（分类 top1 一致 / 检测超 0.3 候选集一致，5 轮随机输入 vs ORT）→ 部署到同目录 Ncnn\。

依赖 `onnx onnxruntime numpy` 与本机 `pnnx.exe`、`yolo_check.exe`（路径可用 `--pnnx`/`--yolo-check` 覆盖）。

背景知识（2026-10-22 踩坑记录）：
- opset 21+（Ultralytics 新导出器）的图谱**节点乱序**且**默认 FP16 导出**（输入被 Cast 成 fp16、
  权重/bias 全 fp16）——前者被 checker 拒收，后者 pnnx 无法折叠 Conv（透传成 ncnn 不认识的通用层）。
  脚本的两步修复即为此设计；q_classify（opset 19、fp32 导出）则天然免修。
- yolo_check/shim 的 `out.total()` 含通道对齐填充（如 w=6 报 8），真实元素数 = c×h×w；
  分类输出按 `names.Length` 截取即可，检测形状全部自然对齐。

## PP-OCRv6 转换流程（已验证）

pnnx 的常量折叠**不支持动态宽度**（`inputshape=[1,3,48,-1]` 会失败），因此按 BetterGI 的桶宽
各转一个固定形状模型，与现有分桶策略天然对齐：

```bat
:: 对每个桶宽 160/320/480/640/800/960 各执行一次
pnnx.exe slim.onnx inputshape="[1,3,48,<宽度>]"
:: 产出 slim.ncnn.param / slim.ncnn.bin → 重命名为 rec_w<宽度>.param / .bin
```

字典必须与 DML 路线同源（yml character_dict 实际 18708 条，手工提取易漏 1 条导致整体错位）：

```bat
OcrDmlProbe.exe --dump-labels --dump-labels-file <模型目录>\dict.txt
```

实测（B580，C# 全流程含归一化+推理+CTC 解码）：

| 桶宽 | ncnn Vulkan | DML 对照 |
|---|---|---|
| 160 | 3.7ms（C++）/ — | ~6ms |
| 320 | 8.6ms | ~20ms |
| 480 | 10.5ms | ~22ms |
| 960 | 8.0ms（C++ 纯推理） | ~22ms |

识别文本与 DML 路线逐字一致（`领取「每日委托」奖励`）。

## 生产接线（已实现）

推理引擎设置页顶部二选一 `ncnn` / `ONNX Runtime`，ncnn 分支再选计算后端（Vulkan / CPU）与具体设备：

- 配置字段：`preferredBackend`（全局意图）、`useNcnnOcr` / `useNcnnYolo`（逐用途生效）、
  `ncnnComputeMode`（`vulkan` / `cpu`）、`ncnnGpuIndex`、`ncnnDetTuning` / `ncnnRecTuning` / `ncnnYoloTuning`
  （三套独立调优参数，逐字段写入 JSON 白名单，缺省回落默认值）、`ncnnThreads`（旧字段，仅作迁移基线）。
  切换全局引擎会重置两个用途开关，高级区可单独覆盖。
- 设置页把 ncnn 整体抽成独立子控件 `NcnnSettingsView` + `NcnnSettingsViewModel`，页面只保留引擎二选一；
  子控件内分四个折叠区：基础（运行库 / 计算后端 / 设备 / 三套线程，默认展开）、精度与调优（三套参数各一子折叠）、
  设备详情与显存（`shim_get_device_info` + `shim_get_heap_budget`）、模型自检（产物清单 + blob 名探测）。
  测速统一走页面底部原有的「性能测试」，ncnn 区不再重复提供测速与耗时统计。
- 计算后端：Vulkan 传 `gpuIndex = ncnnGpuIndex`；CPU 传 `gpuIndex = -1`，由各用途的 `threads` 控制并行度。
- 设备显示：`NcnnShim.QueryDevices()` 调用 `shim_get_device_count/name`，下拉展示真实名称与独显/核显；
  仅在选中 ncnn + Vulkan 时初始化（Vulkan 实例随进程常驻）。枚举失败直接提示，不再退回设备索引。
- 资产目录：`Assets\Model\PaddleOCR\Ncnn\`（`ncnn_shim.dll` + `det_w{320,640,960}.param/bin`
  + `rec_w{160,320,480,640,800,960}.param/bin`，约 93MB）。开关开启后产物缺失或创建失败直接抛
  `NcnnUnavailableException`（消息含 shim 路径、模型目录与首个失败原因），不再静默换用 ONNX；
  启动准备阶段 `InferenceRuntimeService.PrepareStartup` 会先校验运行库并把问题写进启动提示。
- 字典复用 ONNX 路线同一份 `PaddleOcrModelType.RecLabel()`（yml 解析，18708 条），不额外维护 dict 文件。
- 后端抽象：`IOcrDetector` / `IOcrRecognizer`（internal），`Det`/`Rec` 与 `NcnnDet`/`NcnnRec` 各实现一套，
  `PaddleOcrService` 注入任意组合、预热/解码/后处理完全复用。
- 验证：`NcnnShimExTest`（blob 名、张量元信息、内置默认值与 shim 默认一致，CPU 路线默认运行）、
  `NcnnRecTest`（w320/w480 文本逐字一致，8.2~10.0ms）、`NcnnDetTest`（5 框与 ONNX 一致，p50 11.9ms）
  ——后两者走 Vulkan，需 `BETTERGI_NCNN_VULKAN_TESTS=1` 才执行。

## YOLO 模型（ncnn Vulkan）实测

pnnx 可直接转换项目 YOLO 模型（固定形状），B580 实测：

| 模型 | 输入 | ONNX CPU | ncnn Vulkan | 倍数 |
|---|---|---|---|---|
| BgiWorld | 1280×1280 | 90.3ms | **8.27ms** | 10.9× |

数值对拍（`yolo_check.exe` + onnxruntime 同输入比较）：
- 关闭 FP16：最大差 0.284、平均差 0.0042（Conv 算法/累加顺序差异，对框坐标 <0.03%）
- 默认 FP16：平均差 0.066（坐标量级 1000+ 时为 FP16 量化误差）

已知问题：
- `BgiTree` 模型（opset 22）节点非拓扑排序且缺少输入 `graph_input_cast_0`，pnnx 转换崩溃；
  该模型在 ORT 下也从未真正推理过（验证流程实际走 OCR 分支），接入前需重新导出。
- 接入生产还需：C# 侧 YOLOv8 decode + NMS（YoloSharp 绑定 ORT，不可复用其后处理）、
  结果类型隔离（7 处 `Detect` 调用依赖 YoloSharp 类型）。
