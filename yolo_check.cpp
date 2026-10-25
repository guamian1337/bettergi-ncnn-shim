// YOLO 数值对拍：读入 input.bin（NCHW float），ncnn 推理，dump 输出供 Python 比对
// 用法: yolo_check.exe <param> <bin> <input.bin> <W> <H> <out_ncnn.bin> [inName=in0] [outName=out0]
#include <net.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 7) { fprintf(stderr, "用法: yolo_check <param> <bin> <input.bin> <W> <H> <out.bin> [in] [out]\n"); return 1; }
    const char* paramPath = argv[1];
    const char* modelPath = argv[2];
    const char* inputPath = argv[3];
    const int width = atoi(argv[4]);
    const int height = atoi(argv[5]);
    const char* outPath = argv[6];
    const char* inputName = argc > 7 ? argv[7] : "in0";
    const char* outputName = argc > 8 ? argv[8] : "out0";

    FILE* f = fopen(inputPath, "rb");
    if (!f) { fprintf(stderr, "无法打开输入 %s\n", inputPath); return 1; }
    const size_t elems = (size_t)3 * height * width;
    std::vector<float> data(elems);
    if (fread(data.data(), sizeof(float), elems, f) != elems) { fprintf(stderr, "输入尺寸不匹配，期望 %zu\n", elems); fclose(f); return 1; }
    fclose(f);

    ncnn::Net net;
    net.opt.use_vulkan_compute = true;
    net.opt.num_threads = 4;
    if (getenv("NCNN_NO_FP16")) { net.opt.use_fp16_packed = false; net.opt.use_fp16_storage = false; net.opt.use_fp16_arithmetic = false; }
    if (net.load_param(paramPath) != 0) { fprintf(stderr, "load_param 失败\n"); return 1; }
    if (net.load_model(modelPath) != 0) { fprintf(stderr, "load_model 失败\n"); return 1; }

    // NCHW → ncnn Mat(w,h,c)：ncnn 数据按 channel 平面存储，与 NCHW 顺序一致
    ncnn::Mat inp(width, height, 3);
    memcpy(inp.data, data.data(), elems * sizeof(float));

    ncnn::Extractor ex = net.create_extractor();
    if (ex.input(inputName, inp) != 0) { fprintf(stderr, "input 失败\n"); return 1; }
    ncnn::Mat out;
    if (ex.extract(outputName, out) != 0) { fprintf(stderr, "extract 失败\n"); return 1; }

    FILE* g = fopen(outPath, "wb");
    if (!g) { fprintf(stderr, "无法写输出\n"); return 1; }
    fwrite(out.data, sizeof(float), out.total(), g);
    fclose(g);
    printf("输出 c=%d h=%d w=%d 共 %zu 元素\n", out.c, out.h, out.w, out.total());
    const float* p = (const float*)out.data;
    printf("前8值: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
        p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
    return 0;
}
