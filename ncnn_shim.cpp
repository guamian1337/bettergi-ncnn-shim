// ncnn_shim.cpp — 给 C# P/Invoke 用的 C API（加载 PP-OCR / YOLO 的 ncnn 模型并推理）
// 编译：/LD 静态链接 ncnn（Vulkan），导出纯 C 接口
//
// 接口分两代：
//   旧接口 shim_create / shim_infer：行为保持不变，供旧版 C# 降级使用；
//   新接口 shim_create_ex / shim_infer_ex：版本化选项结构、完整 Option 调优、多输入多输出、
//   输出张量元信息（dims/w/h/d/c/elempack/elembits/cstep）、像素直输、blob 名查询、设备详情。
#include <net.h>
#include <gpu.h>
#include <cstring>
#include <cstdlib>
#include <cstddef>

// ---------------- 版本化结构（与 C# NcnnInterop.cs 逐字段对应） ----------------

struct ShimNetOptions
{
    int size;
    const char* paramPath;
    const char* modelPath;
    const char* paramMemory; // 非空时优先 load_param_mem
    int gpuIndex;            // < 0 为纯 CPU 推理
    int numThreads;          // <= 0 使用 ncnn 默认
    int openmpBlocktime;     // < 0 继承默认
    int useWinogradConvolution;
    int useSgemmConvolution;
    int useInt8Inference;
    int useBf16Storage;
    int useFp16Packed;
    int useFp16Storage;
    int useFp16Arithmetic;
    int useInt8Packed;
    int useInt8Storage;
    int useInt8Arithmetic;
    int usePackingLayout;
    int useBf16Packed;
    int useShaderLocalMemory;
    int useCooperativeMatrix;
    int useWinograd23;
    int useWinograd43;
    int useWinograd63;
    int useA53A55OptimizedKernel;
    int useFp16Uniform;
    int useInt8Uniform;
    int useInt16Packed;
    int useInt16Storage;
    int useLocalPoolAllocator;
    int flushDenormals;      // < 0 继承，0-3 见 ncnn Option
};

// pixelFormat: 0 float CHW / 1 RGB888 / 2 BGR888 / 3 RGBA8888 / 4 BGRA8888 / 5 NV12
struct ShimInput
{
    const char* name;
    const void* data;
    int w;
    int h;
    int c;
    int pixelFormat;
    int stride; // 像素行字节数，0 表示按格式推导
};

struct ShimTensor
{
    float* data;
    int dims;
    int w;
    int h;
    int d;
    int c;
    int elempack;
    int elembits;
    int cstep;
    int count;
};

struct ShimExtractorOptions
{
    int size;
    int numThreads; // <= 0 继承
    int lightMode;  // < 0 继承
};

struct ShimDeviceInfo
{
    int size;
    char name[256];
    char driverName[256];
    int type;
    unsigned int vendorId;
    unsigned int deviceId;
    unsigned int apiVersion;
    unsigned int driverVersion;
    unsigned int driverId;
    unsigned int computeQueueCount;
    unsigned int transferQueueCount;
    int unifiedComputeTransferQueue;
    int resizableBarEnabled;
    int supportImageStorage;
    int preferShaderLocalMemory;
    int supportFp16Packed;
    int supportFp16Storage;
    int supportFp16Uniform;
    int supportFp16Arithmetic;
    int supportInt8Packed;
    int supportInt8Storage;
    int supportInt8Uniform;
    int supportInt8Arithmetic;
    int supportInt16Packed;
    int supportInt16Storage;
    int supportInt16Arithmetic;
    int supportBf16Packed;
    int supportBf16Storage;
    int supportFp16Image;
    int supportInt8Image;
    int supportFpFfastMath;
    int supportSubgroupSizeControl;
    int supportComputeFullSubgroups;
    unsigned int supportSubgroupOps;
    unsigned int subgroupSize;
    unsigned int minSubgroupSize;
    unsigned int maxSubgroupSize;
    unsigned int maxSharedMemorySize;
    unsigned int maxWorkgroupInvocations;
    unsigned int roughScore;
    unsigned int heapBudgetMB;
};

namespace {

ncnn::Net* AsNet(void* handle)
{
    return static_cast<ncnn::Net*>(handle);
}

void ApplyOptions(ncnn::Net* net, const ShimNetOptions* o)
{
    ncnn::Option& opt = net->opt;
    if (o->numThreads > 0) opt.num_threads = o->numThreads;
    if (o->openmpBlocktime >= 0) opt.openmp_blocktime = o->openmpBlocktime;

    opt.use_winograd_convolution = o->useWinogradConvolution != 0;
    opt.use_sgemm_convolution = o->useSgemmConvolution != 0;
    opt.use_int8_inference = o->useInt8Inference != 0;
    opt.use_bf16_storage = o->useBf16Storage != 0;
    opt.use_fp16_packed = o->useFp16Packed != 0;
    opt.use_fp16_storage = o->useFp16Storage != 0;
    opt.use_fp16_arithmetic = o->useFp16Arithmetic != 0;
    opt.use_int8_packed = o->useInt8Packed != 0;
    opt.use_int8_storage = o->useInt8Storage != 0;
    opt.use_int8_arithmetic = o->useInt8Arithmetic != 0;
    opt.use_packing_layout = o->usePackingLayout != 0;
    opt.use_bf16_packed = o->useBf16Packed != 0;
    opt.use_shader_local_memory = o->useShaderLocalMemory != 0;
    opt.use_cooperative_matrix = o->useCooperativeMatrix != 0;
    opt.use_winograd23_convolution = o->useWinograd23 != 0;
    opt.use_winograd43_convolution = o->useWinograd43 != 0;
    opt.use_winograd63_convolution = o->useWinograd63 != 0;
    opt.use_a53_a55_optimized_kernel = o->useA53A55OptimizedKernel != 0;
    opt.use_fp16_uniform = o->useFp16Uniform != 0;
    opt.use_int8_uniform = o->useInt8Uniform != 0;
    opt.use_int16_packed = o->useInt16Packed != 0;
    opt.use_int16_storage = o->useInt16Storage != 0;
    opt.use_local_pool_allocator = o->useLocalPoolAllocator != 0;
    if (o->flushDenormals >= 0 && o->flushDenormals <= 3)
        opt.flush_denormals = (unsigned char)o->flushDenormals;

    // Vulkan 必须在 load_param 之前决定，否则层不会创建 GPU 管线。
    opt.use_vulkan_compute = o->gpuIndex >= 0;
    if (o->gpuIndex >= 0) net->set_vulkan_device(o->gpuIndex);
}

ncnn::Mat MakeInput(const ShimInput& in)
{
    if (in.pixelFormat == 0)
    {
        const int c = in.c > 0 ? in.c : 1;
        return ncnn::Mat(in.w, in.h, c, (void*)in.data);
    }

    const unsigned char* pixels = static_cast<const unsigned char*>(in.data);

    if (in.pixelFormat == 5)
    {
        // NV12：ncnn 没有对应 from_pixels 类型，先按目标尺寸转 RGB888 再走 from_pixels。
        ncnn::Mat rgb(in.w, in.h, (size_t)3u, 1);
        if (rgb.empty()) return ncnn::Mat();
        ncnn::resize_bilinear_yuv420sp(pixels, in.w, in.h,
                                       static_cast<unsigned char*>(rgb.data), in.w, in.h);
        return ncnn::Mat::from_pixels(static_cast<const unsigned char*>(rgb.data),
                                      ncnn::Mat::PIXEL_RGB, in.w, in.h);
    }

    int type = ncnn::Mat::PIXEL_RGB;
    switch (in.pixelFormat)
    {
    case 2: type = ncnn::Mat::PIXEL_BGR; break;
    case 3: type = ncnn::Mat::PIXEL_RGBA; break;
    case 4: type = ncnn::Mat::PIXEL_BGRA; break;
    default: type = ncnn::Mat::PIXEL_RGB; break;
    }

    if (in.stride > 0)
        return ncnn::Mat::from_pixels(pixels, type, in.w, in.h, in.stride);
    return ncnn::Mat::from_pixels(pixels, type, in.w, in.h);
}

static bool EnsureGpuInstance()
{
    static const bool ready = ncnn::create_gpu_instance() == 0;
    return ready;
}

} // namespace

extern "C" {

// ---------------- 旧接口（行为保持不变） ----------------

__declspec(dllexport) void* shim_create(const char* paramPath, const char* modelPath,
                                        int gpuIndex, int numThreads)
{
    auto* net = new (std::nothrow) ncnn::Net();
    if (!net) return nullptr;
    net->opt.use_vulkan_compute = gpuIndex >= 0;
    net->opt.num_threads = numThreads;
    if (gpuIndex >= 0) net->set_vulkan_device(gpuIndex);
    if (net->load_param(paramPath) != 0) { delete net; return nullptr; }
    if (net->load_model(modelPath) != 0) { delete net; return nullptr; }
    return net;
}

__declspec(dllexport) void shim_destroy(void* handle)
{
    delete AsNet(handle);
}

__declspec(dllexport) void shim_free_buffer(void* p)
{
    free(p);
}

// ---------------- 新接口：创建会话 ----------------

__declspec(dllexport) void* shim_create_ex(const ShimNetOptions* options)
{
    if (!options || options->size < (int)sizeof(ShimNetOptions)) return nullptr;

    auto* net = new (std::nothrow) ncnn::Net();
    if (!net) return nullptr;

    ApplyOptions(net, options);

    int rc = options->paramMemory
                 ? net->load_param_mem(options->paramMemory)
                 : net->load_param(options->paramPath);
    if (rc != 0) { delete net; return nullptr; }
    if (net->load_model(options->modelPath) != 0) { delete net; return nullptr; }
    return net;
}

// 返回 ncnn::Option 的默认快照，供 C# 侧把“未改动”表达成真实默认值。
__declspec(dllexport) void shim_get_default_options(ShimNetOptions* out)
{
    if (!out || out->size < (int)sizeof(ShimNetOptions)) return;

    const int size = out->size;
    memset(out, 0, sizeof(ShimNetOptions));
    out->size = size;

    ncnn::Option opt;
    out->paramPath = nullptr;
    out->modelPath = nullptr;
    out->paramMemory = nullptr;
    out->gpuIndex = -1;
    out->numThreads = opt.num_threads;
    out->openmpBlocktime = opt.openmp_blocktime;
    out->useWinogradConvolution = opt.use_winograd_convolution ? 1 : 0;
    out->useSgemmConvolution = opt.use_sgemm_convolution ? 1 : 0;
    out->useInt8Inference = opt.use_int8_inference ? 1 : 0;
    out->useBf16Storage = opt.use_bf16_storage ? 1 : 0;
    out->useFp16Packed = opt.use_fp16_packed ? 1 : 0;
    out->useFp16Storage = opt.use_fp16_storage ? 1 : 0;
    out->useFp16Arithmetic = opt.use_fp16_arithmetic ? 1 : 0;
    out->useInt8Packed = opt.use_int8_packed ? 1 : 0;
    out->useInt8Storage = opt.use_int8_storage ? 1 : 0;
    out->useInt8Arithmetic = opt.use_int8_arithmetic ? 1 : 0;
    out->usePackingLayout = opt.use_packing_layout ? 1 : 0;
    out->useBf16Packed = opt.use_bf16_packed ? 1 : 0;
    out->useShaderLocalMemory = opt.use_shader_local_memory ? 1 : 0;
    out->useCooperativeMatrix = opt.use_cooperative_matrix ? 1 : 0;
    out->useWinograd23 = opt.use_winograd23_convolution ? 1 : 0;
    out->useWinograd43 = opt.use_winograd43_convolution ? 1 : 0;
    out->useWinograd63 = opt.use_winograd63_convolution ? 1 : 0;
    out->useA53A55OptimizedKernel = opt.use_a53_a55_optimized_kernel ? 1 : 0;
    out->useFp16Uniform = opt.use_fp16_uniform ? 1 : 0;
    out->useInt8Uniform = opt.use_int8_uniform ? 1 : 0;
    out->useInt16Packed = opt.use_int16_packed ? 1 : 0;
    out->useInt16Storage = opt.use_int16_storage ? 1 : 0;
    out->useLocalPoolAllocator = opt.use_local_pool_allocator ? 1 : 0;
    out->flushDenormals = opt.flush_denormals;
}

// ---------------- blob 名查询 ----------------

__declspec(dllexport) int shim_get_input_count(void* handle)
{
    if (!handle) return -1;
    return (int)AsNet(handle)->input_names().size();
}

__declspec(dllexport) int shim_get_output_count(void* handle)
{
    if (!handle) return -1;
    return (int)AsNet(handle)->output_names().size();
}

__declspec(dllexport) int shim_get_input_name(void* handle, int index, char* buffer, int bufferSize)
{
    if (!handle || !buffer || bufferSize <= 0) return -1;
    buffer[0] = '\0';
    const auto& names = AsNet(handle)->input_names();
    if (index < 0 || index >= (int)names.size()) return -1;
    strncpy_s(buffer, (size_t)bufferSize, names[index] ? names[index] : "", _TRUNCATE);
    return 0;
}

__declspec(dllexport) int shim_get_output_name(void* handle, int index, char* buffer, int bufferSize)
{
    if (!handle || !buffer || bufferSize <= 0) return -1;
    buffer[0] = '\0';
    const auto& names = AsNet(handle)->output_names();
    if (index < 0 || index >= (int)names.size()) return -1;
    strncpy_s(buffer, (size_t)bufferSize, names[index] ? names[index] : "", _TRUNCATE);
    return 0;
}

// ---------------- 推理 ----------------

__declspec(dllexport) int shim_infer_ex(void* handle,
                                        const ShimInput* inputs, int inputCount,
                                        const char* const* outputNames, int outputCount,
                                        const ShimExtractorOptions* extractorOptions,
                                        ShimTensor* outputs)
{
    if (!handle || !inputs || inputCount <= 0 || !outputNames || outputCount <= 0 || !outputs)
        return -1;

    auto* net = AsNet(handle);
    ncnn::Extractor ex = net->create_extractor();
    if (extractorOptions && extractorOptions->size >= (int)sizeof(ShimExtractorOptions))
    {
        // 新版 ncnn 的 Extractor 不再提供 set_num_threads，线程数统一由 Net Option 决定；
        // 字段保留在结构体里以便未来 ncnn 恢复该能力时不需要改 ABI。
        if (extractorOptions->lightMode >= 0) ex.set_light_mode(extractorOptions->lightMode != 0);
    }

    for (int i = 0; i < inputCount; i++)
    {
        ncnn::Mat in = MakeInput(inputs[i]);
        if (in.empty()) return -2;
        if (ex.input(inputs[i].name, in) != 0) return -3;
    }

    for (int i = 0; i < outputCount; i++)
    {
        ncnn::Mat out;
        // type = 0：把 fp16/bf16 与 packing 统一转回 fp32 unpacked。
        if (ex.extract(outputNames[i], out, 0) != 0) return -4;
        if (out.elembits() != 32 || out.elempack != 1) return -6;

        const size_t perChannel = (size_t)out.w * out.h * (out.d > 1 ? out.d : 1);
        const size_t total = perChannel * out.c;
        auto* buffer = static_cast<float*>(malloc(total * sizeof(float)));
        if (!buffer) return -5;

        size_t offset = 0;
        for (int q = 0; q < out.c; q++)
        {
            memcpy(buffer + offset, out.channel(q), perChannel * sizeof(float));
            offset += perChannel;
        }

        ShimTensor& t = outputs[i];
        memset(&t, 0, sizeof(t));
        t.data = buffer;
        t.dims = out.dims;
        t.w = out.w;
        t.h = out.h;
        t.d = out.d;
        t.c = out.c;
        t.elempack = out.elempack;
        t.elembits = out.elembits();
        t.cstep = (int)out.cstep;
        t.count = (int)total;
    }

    return 0;
}

// 旧接口：单输入单输出 float CHW，内部转调新接口。
__declspec(dllexport) int shim_infer(void* handle, const char* inputName, const char* outputName,
                                     const float* data, int w, int h, int c,
                                     float** outBuffer, int* outCount)
{
    if (!outBuffer || !outCount) return -1;
    ShimInput input{};
    input.name = inputName;
    input.data = data;
    input.w = w;
    input.h = h;
    input.c = c;
    input.pixelFormat = 0;
    input.stride = 0;

    const char* names[1] = { outputName };
    ShimTensor tensor{};
    const int rc = shim_infer_ex(handle, &input, 1, names, 1, nullptr, &tensor);
    if (rc != 0) return rc;
    *outBuffer = tensor.data;
    *outCount = tensor.count;
    return 0;
}

// ---------------- 像素直输 ----------------

__declspec(dllexport) int shim_infer_pixels(void* handle,
                                            const unsigned char* pixels, int pixelFormat,
                                            int w, int h, int stride,
                                            const char* inputName, const char* outputName,
                                            ShimTensor* output)
{
    if (!output) return -1;
    ShimInput input{};
    input.name = inputName;
    input.data = pixels;
    input.w = w;
    input.h = h;
    input.c = 3;
    input.pixelFormat = pixelFormat;
    input.stride = stride;

    const char* names[1] = { outputName };
    return shim_infer_ex(handle, &input, 1, names, 1, nullptr, output);
}

// ---------------- 设备枚举与详情 ----------------

__declspec(dllexport) int shim_get_device_count()
{
    return EnsureGpuInstance() ? ncnn::get_gpu_count() : 0;
}

__declspec(dllexport) int shim_get_default_device()
{
    if (!EnsureGpuInstance()) return -1;
    return ncnn::get_gpu_count() > 0 ? ncnn::get_default_gpu_index() : -1;
}

__declspec(dllexport) int shim_get_device_name(int index, char* buffer, int bufferSize)
{
    if (!buffer || bufferSize <= 0) return -1;
    buffer[0] = '\0';
    if (!EnsureGpuInstance()) return -1;
    if (index < 0 || index >= ncnn::get_gpu_count()) return -1;

    const ncnn::GpuInfo& info = ncnn::get_gpu_info(index);
    const char* name = info.device_name();
    if (name) strncpy_s(buffer, (size_t)bufferSize, name, _TRUNCATE);
    return info.type();
}

__declspec(dllexport) int shim_get_device_info(int index, ShimDeviceInfo* info)
{
    if (!info) return -1;
    const int size = info->size;
    memset(info, 0, sizeof(ShimDeviceInfo));
    info->size = size;
    if (!EnsureGpuInstance()) return -2;
    if (index < 0 || index >= ncnn::get_gpu_count()) return -3;

    const ncnn::GpuInfo& g = ncnn::get_gpu_info(index);
    const char* name = g.device_name();
    const char* driverName = g.driver_name();
    if (name) strncpy_s(info->name, sizeof(info->name), name, _TRUNCATE);
    if (driverName) strncpy_s(info->driverName, sizeof(info->driverName), driverName, _TRUNCATE);

    info->type = g.type();
    info->vendorId = g.vendor_id();
    info->deviceId = g.device_id();
    info->apiVersion = g.api_version();
    info->driverVersion = g.driver_version();
    info->driverId = g.driver_id();
    info->computeQueueCount = g.compute_queue_count();
    info->transferQueueCount = g.transfer_queue_count();
    info->unifiedComputeTransferQueue = g.unified_compute_transfer_queue() ? 1 : 0;
    info->resizableBarEnabled = g.resizable_bar_enabled() ? 1 : 0;
    info->supportImageStorage = g.support_image_storage() ? 1 : 0;
    info->preferShaderLocalMemory = g.prefer_shader_local_memory() ? 1 : 0;
    info->supportFp16Packed = g.support_fp16_packed() ? 1 : 0;
    info->supportFp16Storage = g.support_fp16_storage() ? 1 : 0;
    info->supportFp16Uniform = g.support_fp16_uniform() ? 1 : 0;
    info->supportFp16Arithmetic = g.support_fp16_arithmetic() ? 1 : 0;
    info->supportInt8Packed = g.support_int8_packed() ? 1 : 0;
    info->supportInt8Storage = g.support_int8_storage() ? 1 : 0;
    info->supportInt8Uniform = g.support_int8_uniform() ? 1 : 0;
    info->supportInt8Arithmetic = g.support_int8_arithmetic() ? 1 : 0;
    info->supportInt16Packed = g.support_int16_packed() ? 1 : 0;
    info->supportInt16Storage = g.support_int16_storage() ? 1 : 0;
    info->supportInt16Arithmetic = g.support_int16_arithmetic() ? 1 : 0;
    info->supportBf16Packed = g.support_bf16_packed() ? 1 : 0;
    info->supportBf16Storage = g.support_bf16_storage() ? 1 : 0;
    info->supportFp16Image = g.support_fp16_image() ? 1 : 0;
    info->supportInt8Image = g.support_int8_image() ? 1 : 0;
    info->supportFpFfastMath = g.support_fp_fast_math() ? 1 : 0;
    info->supportSubgroupSizeControl = g.support_subgroup_size_control() ? 1 : 0;
    info->supportComputeFullSubgroups = g.support_compute_full_subgroups() ? 1 : 0;
    info->supportSubgroupOps = g.support_subgroup_ops();
    info->subgroupSize = g.subgroup_size();
    info->minSubgroupSize = g.min_subgroup_size();
    info->maxSubgroupSize = g.max_subgroup_size();
    info->maxSharedMemorySize = g.max_shared_memory_size();
    info->maxWorkgroupInvocations = g.max_workgroup_invocations();
    info->roughScore = g.rough_score();
    // heap budget 在 VulkanDevice 上，GpuInfo 只提供静态能力信息。
    const ncnn::VulkanDevice* device = ncnn::get_gpu_device(index);
    info->heapBudgetMB = device ? device->get_heap_budget() : 0;

    return 0;
}

__declspec(dllexport) unsigned int shim_get_heap_budget(int index)
{
    if (!EnsureGpuInstance()) return 0;
    if (index < 0 || index >= ncnn::get_gpu_count()) return 0;
    const ncnn::VulkanDevice* device = ncnn::get_gpu_device(index);
    return device ? device->get_heap_budget() : 0;
}

} // extern "C"
