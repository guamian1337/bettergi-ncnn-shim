// ncnn Vulkan 基准：PP-OCRv3 rec 模型在 B580 上的单次推理耗时
// 用法: ocr_rec_bench.exe <rec.param> <rec.bin> [宽=480] [高=48] [轮数=30]
#include <net.h>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>

int main(int argc, char** argv)
{
    const char* paramPath = argc > 1 ? argv[1] : "ch_PP-OCRv3_rec.param";
    const char* modelPath = argc > 2 ? argv[2] : "ch_PP-OCRv3_rec.bin";
    const int width = argc > 3 ? atoi(argv[3]) : 480;
    const int height = argc > 4 ? atoi(argv[4]) : 48;
    const int iters = argc > 5 ? atoi(argv[5]) : 30;

    ncnn::Net rec;
    rec.opt.use_vulkan_compute = true;
    rec.opt.num_threads = 4;
    if (rec.load_param(paramPath) != 0) { fprintf(stderr, "load_param 失败\n"); return 1; }
    if (rec.load_model(modelPath) != 0) { fprintf(stderr, "load_model 失败\n"); return 1; }
    fprintf(stderr, "vulkan 设备数: %d\n", ncnn::get_gpu_count());

    // 固定输入内容（模拟归一化后的特征图数据）
    ncnn::Mat in(width, height, 3);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (int c = 0; c < 3; c++)
        for (int h = 0; h < height; h++)
            for (int w = 0; w < width; w++)
                in.channel(c).row(h)[w] = dist(rng);

    // 预热（含各算子 pipeline 编译与内存分配）
    for (int w = 0; w < 8; w++)
    {
        ncnn::Extractor ex = rec.create_extractor();
        ex.input("input", in);
        ncnn::Mat out;
        if (ex.extract("out", out) != 0) { fprintf(stderr, "extract 失败\n"); return 1; }
    }

    double total = 0, minT = 1e9, maxT = 0;
    size_t outElems = 0;
    for (int i = 0; i < iters; i++)
    {
        auto t0 = std::chrono::steady_clock::now();
        ncnn::Extractor ex = rec.create_extractor();
        ex.input("input", in);
        ncnn::Mat out;
        if (ex.extract("out", out) != 0) { fprintf(stderr, "extract 失败\n"); return 1; }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total += ms;
        if (ms < minT) minT = ms;
        if (ms > maxT) maxT = ms;
        outElems = out.total();
        printf("#%3d  %7.2fms  输出 %d x %d x %d\n", i, ms, out.c, out.h, out.w);
    }
    printf("==== rec %dx%d, %d 次 ====\n", width, height, iters);
    printf("平均 %.2fms  最快 %.2fms  最慢 %.2fms\n", total / iters, minT, maxT);
    printf("输出元素数: %zu\n", outElems);
    return 0;
}
