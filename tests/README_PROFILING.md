# Module Latency Profiling for FastVGGT

## 概述

本脚本用于测量FastVGGT模型在不同帧数下不同模块的耗时，以识别模型的性能瓶颈并评估Token Merging的加速效果。

## 使用方法

### 1. 使用Dummy数据（快速测试）
```bash
python tests/measure_module_latency.py --dataset_type dummy
```

### 2. 使用7Scenes数据集
```bash
python tests/measure_module_latency.py \
    --dataset_type 7scenes \
    --data_dir /path/to/7scenes \
    --resolution 518 392
```

### 3. 使用ScanNet数据集
```bash
python tests/measure_module_latency.py \
    --dataset_type scannet \
    --data_dir /path/to/scannet
```

### 4. 使用自定义图像目录
```bash
python tests/measure_module_latency.py \
    --dataset_type images \
    --data_dir /path/to/images
```

## 配置参数

编辑脚本顶部的全局变量以自定义测试：

```python
# 测试的帧数列表
FRAME_COUNTS = [5, 10, 20, 30, 50, 100]

# 各帧数对应的批次大小（根据GPU内存调整）
BATCH_SIZES = {
    5: 3,
    10: 2,
    20: 1,
    ...
}

# Token Merge比例：0.9表示启用merge，0.0表示无merge
MERGE_RATIOS = [0.9, 0.0]

# 每个配置的运行次数（用于平均）
NUM_RUNS = 5
```

## 输出结果

结果保存在 `tests_result/module_latency_report.csv` 中，包含以下列：

| 列名 | 说明 | 单位 |
|------|------|------|
| `seq_len` | 请求的序列长度（帧数） | 帧 |
| `actual_frames` | 实际处理的帧数 | 帧 |
| `batch_size` | 批处理大小 | - |
| `merge_ratio` | Token Merge比例 (0.0-1.0) | - |
| `merging_threshold` | Merging阈值 | - |
| `dataset` | 使用的数据集 | - |
| `mode` | 测试模式（with_merge/no_merge） | - |
| `total_time_ms` | 总耗时 | 毫秒 |
| `throughput_fps` | 吞吐量 | 帧/秒 |
| `top1_module` | 耗时最多的模块 | - |
| `top1_time_ms` | 耗时最多模块的耗时 | 毫秒 |
| `top1_percent` | 耗时最多模块的占比 | % |
| `top5_summary` | 耗时TOP5模块的摘要 | - |

## 测量指标解释

### Token冗余性指标

不同的测量指标可以体现不同方面的Token冗余性：

#### 1. **全局注意力的耗时增长** (最重要)
- **为什么重要**: 全局注意力的复杂度为 $O(S^2 \cdot N)$，其中 $S$ 是序列长度，$N$ 是Token数量
- **冗余体现**: 相邻帧间高度相似，导致大量Token冗余
- **观察指标**: 
  - 对比 `with_merge` vs `no_merge` 的全局块耗时
  - 计算加速比：$\text{Speedup} = \frac{\text{耗时\_no\_merge}}{\text{耗时\_with\_merge}}$
  - 帧数越多，加速比应越明显

#### 2. **深度预测头的加速** (密集预测)
- **为什么重要**: 深度/点云预测需要处理每个Token，Token冗余直接影响计算量
- **冗余体现**: 减少Token数量可以减少密集预测层的计算
- **观察指标**: `depth_head`, `point_head` 的耗时随帧数的增长曲线

#### 3. **总体吞吐量 (FPS)**
- **为什么重要**: 最终用户关心的是模型的实际推理速度
- **冗余体现**: Token Merge能提升FPS
- **计算公式**: $\text{FPS} = \frac{\text{帧数}}{\text{耗时(秒)}}$

## 数据精度检查

### 批次大小对齐
- 脚本自动根据 `BATCH_SIZES` 配置调整批次大小
- 实际加载的批次大小可能小于配置值（如果数据不足）
- 请检查 `batch_size` 列确保与预期一致

### 帧数对齐
- 脚本自动处理帧数不足的情况
- 请检查 `actual_frames` 列，应等于或接近 `seq_len`

### OOM（内存不足）处理
- 脚本自动跳过导致OOM的配置
- 日志中会显示 "⚠ OOM during warmup/run"
- 对应行在CSV中不会出现

## 故障排除

### 问题：所有运行都OOM
**解决方案**:
1. 减小FRAME_COUNTS中的值
2. 减小BATCH_SIZES中的值
3. 使用较小的GPU（或确保GPU内存足够）

### 问题：数据集加载失败
**解决方案**:
1. 检查 `--data_dir` 路径是否正确
2. 对于7Scenes，确保目录结构为：`data_dir/scene_name/color/` 和 `data_dir/scene_name/pose/`
3. 对于ScanNet，确保目录结构为：`data_dir/scene_name/color/`

### 问题：模型推理报错
**解决方案**:
1. 检查checkpoint文件是否存在（默认在 `ckpt/model_tracker_fixed_e20.pt`）
2. 确保CUDA可用：`python -c "import torch; print(torch.cuda.is_available())"`

## 结果分析建议

### 1. 生成加速比对比表
```python
import pandas as pd
df = pd.read_csv('tests_result/module_latency_report.csv')

# 对于每个seq_len，计算with_merge vs no_merge的加速比
for seq_len in df['seq_len'].unique():
    with_merge = df[(df['seq_len']==seq_len) & (df['mode']=='with_merge')]['total_time_ms'].values
    no_merge = df[(df['seq_len']==seq_len) & (df['mode']=='no_merge')]['total_time_ms'].values
    if len(with_merge) > 0 and len(no_merge) > 0:
        speedup = no_merge[0] / with_merge[0]
        print(f"Seq {seq_len}: {speedup:.2f}x speedup")
```

### 2. 识别主要瓶颈
检查 `top1_module` 和 `top1_percent` 列，找出耗时最多的模块。

### 3. 评估Token冗余程度
- 帧数越多，加速比越大 → Token冗余程度越高
- 如果 5帧 vs 10帧的加速比差异不大 → Token冗余随帧数非线性增长

## 相关参数说明

### merge_ratio
- **0.9**: 启用Token Merging，保留90%的Token
- **0.0**: 禁用Token Merging，处理所有Token

### merging_threshold
- 在什么Block层数开始启用Token Merging
- **0**: 从第一层开始
- **25**: 从第25层开始（较晚启用）

## 数据集要求

### 7Scenes
- **格式**: 标准7Scenes数据集格式
- **分辨率选项**: (518, 392), (512, 384), (224, 224)
- **要求**: 每个sequence至少3张图片

### ScanNet
- **格式**: 处理后的ScanNet格式，包含color和pose信息
- **分辨率**: 根据数据集自动决定
- **要求**: 每个scene至少3张图片

### Dummy
- **用途**: 快速测试，无需真实数据
- **分辨率**: 默认 (518, 392)
- **特点**: 使用随机数据

## 性能优化建议

1. **减少NUM_RUNS** (如设为1) 以加快测试，但会降低结果可靠性
2. **使用较小的FRAME_COUNTS** 进行快速测试
3. **在多GPU上并行运行** 不同的数据集测试
4. **使用SSD存储** 数据集以加快I/O速度
