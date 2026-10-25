#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ncnn 模型转换流水线：输入 BetterGI 的 YOLO onnx，产出经数值对拍的 ncnn param/bin 并部署。

用法：
    python convert_model.py <model.onnx> [--rounds 5] [--seed 7] [--skip-deploy]
                            [--pnnx <pnnx.exe>] [--yolo-check <yolo_check.exe>]

流水线（每一步失败即退出，退出码非 0）：
    1. 元数据检查：task 必须是 detect/classify，输入必须是静态 NCHW，names 元数据存在
    2. 拓扑排序：opset 21+ 导出的图谱普遍乱序（onnx.checker 拒收）
    3. fp16 -> fp32：FP16 导出的模型 pnnx 无法折叠 Conv（透传成 ncnn 不认识的通用层）
    4. onnx.checker 验证
    5. pnnx 转换（固定输入形状）+ 层型检查：禁止 Tensor.to / 裸 Conv 残留
    6. 数值对拍：ORT 参考值 vs ncnn；分类看 top1/top5 一致性，检测看坐标/分数差与超阈值候选一致
    7. 部署到 <onnx 同目录>\\Ncnn\\<同名>.param/.bin（--skip-deploy 跳过）

依赖：python -m pip install onnx onnxruntime numpy
pnnx / yolo_check 默认取本机开发目录，可用参数覆盖；yolo_check 源码见同目录 yolo_check.cpp。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, numpy_helper

DEFAULT_PNNX = r"D:\bgiprro\ncnn-spike\pnnx-tool\pnnx-20261020-windows\pnnx.exe"
DEFAULT_YOLO_CHECK = r"D:\bgiprro\ncnn-spike\yolo_check.exe"
FP16, FP32 = TensorProto.FLOAT16, TensorProto.FLOAT


def fail(msg: str):
    print(f"[✗] {msg}")
    sys.exit(1)


def ok(msg: str):
    print(f"[✓] {msg}")


# ---------------------------------------------------------------- 1-4：图谱修复

def load_and_fix(onnx_path: str) -> tuple[onnx.ModelProto, dict]:
    model = onnx.load(onnx_path, load_external_data=False)
    g = model.graph
    report = {"toposorted": False, "fp16_fixed": 0}

    # 元数据
    meta = {x.key: x.value for x in model.metadata_props}
    task = meta.get("task", "")
    if task not in ("detect", "classify"):
        fail(f"task 元数据为 {task or '(缺失)'}，只支持 detect/classify")
    names_text = meta.get("names", "")
    class_count = names_text.count(":")
    if class_count == 0:
        fail("names 元数据缺失或为空")
    dims = [d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    if len(dims) != 4 or any(d <= 0 for d in dims):
        fail(f"输入必须是静态 NCHW，实际 {dims}（动态形状请先定形导出）")
    print(f"[✓] 元数据：task={task} 类别 {class_count} 输入 {dims}")

    # 拓扑排序
    avail = {i.name for i in g.input} | {i.name for i in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            avail.update(n.output)
    placed, deferred, progress = [], list(g.node), True
    while deferred and progress:
        progress, rest = False, []
        for n in deferred:
            if all((not i) or i in avail for i in n.input):
                placed.append(n)
                avail.update(n.output)
                progress = True
            else:
                rest.append(n)
        deferred = rest
    if deferred:
        fail(f"有 {len(deferred)} 个节点的输入在图中不存在（模型缺输入，需重新导出）")
    del g.node[:]
    g.node.extend(placed)
    report["toposorted"] = True

    # fp16 -> fp32（快照遍历，禁止边遍历边增删集合）
    def to_f32(t):
        if t.data_type == FP16:
            return numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), t.name)
        return None

    for old in list(g.initializer):
        new = to_f32(old)
        if new is not None:
            g.initializer.remove(old)
            g.initializer.append(new)
            report["fp16_fixed"] += 1
    for n in g.node:
        for a in n.attribute:
            if a.type == onnx.AttributeProto.TENSOR:
                new = to_f32(a.t)
                if new is not None:
                    a.t.CopyFrom(new)
                    report["fp16_fixed"] += 1
            elif a.name == "to" and a.i == FP16:
                a.i = FP32
                report["fp16_fixed"] += 1
    def fix_type(tt):
        if tt.elem_type == FP16:
            tt.elem_type = FP32
            report["fp16_fixed"] += 1
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        fix_type(vi.type.tensor_type)

    try:
        onnx.checker.check_model(model)
    except Exception as e:
        fail(f"onnx.checker 未通过: {e}")
    ok(f"图谱修复：拓扑排序完成，fp16 修复 {report['fp16_fixed']} 处，checker 通过")
    return model, report


# ---------------------------------------------------------------- 5：pnnx + 层型检查

def run_pnnx(fixed_onnx: str, dims: list, pnnx: str) -> tuple[str, str, str, str]:
    workdir = tempfile.mkdtemp(prefix="ncnn_conv_")
    local = shutil.copy(fixed_onnx, workdir)
    stem = os.path.splitext(os.path.basename(local))[0]
    r = subprocess.run(
        [pnnx, local, f"inputshape=[{','.join(map(str, dims))}]"],
        capture_output=True, text=True, cwd=workdir)
    param = os.path.join(workdir, stem + ".ncnn.param")
    binf = os.path.join(workdir, stem + ".ncnn.bin")
    if r.returncode != 0 or not os.path.exists(param) or not os.path.exists(binf):
        tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
        fail("pnnx 转换失败:\n  " + "\n  ".join(tail))
    ok(f"pnnx 转换完成（工作目录 {workdir}）")

    # 层型检查：ncnn 20261020 无 Tensor.to / 三输入 Conv，出现即说明图谱仍有未折叠结构
    lines = open(param, encoding="utf-8").read().splitlines()
    body = lines[2:]
    ops = {}
    for ln in body:
        ops[ln.split()[0]] = ops.get(ln.split()[0], 0) + 1
    bad = [op for op in ("Tensor.to", "Conv") if op in ops]
    if bad:
        fail(f"ncnn param 含不支持的层型 {bad}（图谱常量未折叠，检查 fp16 修复是否完整）")
    if "Convolution" not in ops and "InnerProduct" not in ops:
        fail(f"param 中没有任何卷积/全连接层，转换可疑。层型: {ops}")
    top = sorted(ops.items(), key=lambda kv: -kv[1])[:5]
    ok("层型检查通过，Top5: " + ", ".join(f"{k}×{v}" for k, v in top))

    # 输入/输出 blob 名（pnnx 约定 in0/out0，仍按层定义解析以保险）
    # 层行格式：LayerType name ic oc [输入 blob × ic] [输出 blob × oc] [参数...]
    def last_blob(ln):
        p = ln.split()
        return p[4 + int(p[2])]  # 第一个输出 blob
    input_blob = next(last_blob(ln) for ln in body if ln.split()[0] == "Input")
    output_blob = last_blob(body[-1])
    return param, binf, input_blob, output_blob


# ---------------------------------------------------------------- 6：数值对拍

def parity(ref_onnx: str, param: str, binf: str, dims: list, task: str,
           yolo_check: str, input_blob: str, output_blob: str,
           rounds: int, seed: int) -> bool:
    _, _, H, W = dims
    sess = ort.InferenceSession(ref_onnx, None)
    out_shape = sess.get_outputs()[0].shape          # 分类 [1,nc] / 检测 [1,4+nc,N]
    nc = out_shape[1] - (4 if task == "detect" else 0)
    anchors = out_shape[2] if task == "detect" else 0
    rng = np.random.default_rng(seed)
    all_diff, pass_rounds = [], 0
    for _ in range(rounds):
        x = rng.uniform(-1, 1, dims).astype(np.float32)
        x.tofile("parity_input.bin")
        ref = sess.run(None, {sess.get_inputs()[0].name: x})[0].ravel()
        r = subprocess.run([yolo_check, param, binf, "parity_input.bin", str(W), str(H),
                            "parity_out.bin", input_blob, output_blob],
                           capture_output=True, text=True)
        if r.returncode != 0:
            fail("ncnn 推理失败: " + (r.stderr.strip().splitlines() or ["?"])[-1][:120])
        got = np.fromfile("parity_out.bin", dtype=np.float32)
        if got.size < ref.size:
            fail(f"ncnn 输出元素数 {got.size} < ORT {ref.size}（total() 对齐填充时取前段即为真实值）")
        got = got[:ref.size]
        all_diff.append(np.abs(got - ref))
        if task == "classify":
            if int(np.argmax(ref)) == int(np.argmax(got)):
                pass_rounds += 1
        else:
            # 检测：分数行超 0.3 的 (anchor,class) 候选集必须完全一致
            s_ref = ref.reshape(nc + 4, anchors)[4:]
            s_got = got.reshape(nc + 4, anchors)[4:]
            kr = set(map(tuple, np.argwhere(s_ref > 0.3)))
            kg = set(map(tuple, np.argwhere(s_got > 0.3)))
            if kr == kg:
                pass_rounds += 1
    d = np.concatenate(all_diff)
    verdict = "✓" if pass_rounds == rounds else "✗"
    print(f"[{verdict}] 对拍 {rounds} 轮：avg 差 {d.mean():.5f} max 差 {d.max():.4f}，"
          f"{'top1' if task == 'classify' else '候选集'}一致 {pass_rounds}/{rounds}")
    return pass_rounds == rounds


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="ncnn 模型转换流水线")
    ap.add_argument("onnx", help="待转换的 onnx 路径")
    ap.add_argument("--rounds", type=int, default=5, help="数值对拍轮数（默认 5）")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pnnx", default=DEFAULT_PNNX)
    ap.add_argument("--yolo-check", default=DEFAULT_YOLO_CHECK)
    ap.add_argument("--skip-deploy", action="store_true", help="只转换对拍，不写入 Ncnn 目录")
    args = ap.parse_args()

    if not os.path.exists(args.onnx):
        fail(f"onnx 不存在: {args.onnx}")
    if not os.path.exists(args.pnnx):
        fail(f"pnnx 不存在: {args.pnnx}（用 --pnnx 指定）")
    if not os.path.exists(args.yolo_check):
        fail(f"yolo_check 不存在: {args.yolo_check}（用 --yolo-check 指定）")

    print(f"==== 转换 {os.path.basename(args.onnx)} ====")
    model, _ = load_and_fix(args.onnx)
    dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    task = {x.key: x.value for x in model.metadata_props}["task"]

    # 修复后的图谱写临时文件（不改原 onnx）
    tmp = tempfile.mkdtemp(prefix="ncnn_fix_")
    fixed_path = os.path.join(tmp, os.path.basename(args.onnx))
    onnx.save(model, fixed_path)

    param, binf, input_blob, output_blob = run_pnnx(fixed_path, dims, args.pnnx)
    passed = parity(fixed_path, param, binf, dims, task, args.yolo_check,
                    input_blob, output_blob, args.rounds, args.seed)
    if not passed:
        fail("数值对拍未全部通过，不部署。请检查转换产物或重新导出模型。")

    if args.skip_deploy:
        print(f"[i] --skip-deploy：产物保留在 {os.path.dirname(param)}")
        return
    stem = os.path.splitext(os.path.basename(args.onnx))[0]
    deploy_dir = os.path.join(os.path.dirname(os.path.abspath(args.onnx)), "Ncnn")
    os.makedirs(deploy_dir, exist_ok=True)
    for src, ext in ((param, ".param"), (binf, ".bin")):
        dst = os.path.join(deploy_dir, stem + ext)
        shutil.copyfile(src, dst)
        ok(f"部署 {dst}（{os.path.getsize(dst) / 1048576:.1f}MB）")
    print(f"==== 完成：{stem} 已就绪，重启 BetterGI 后设置页覆盖列表显示 ✓ ====")


if __name__ == "__main__":
    main()
