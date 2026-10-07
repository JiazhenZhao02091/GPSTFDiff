# 指标计算约定

核心公式参考 [prowDIY/STF](https://github.com/prowDIY/STF/tree/main/src/metrics)，并补充 mask 和数值边界处理。

所有指标使用 `metric(gt, pred, mask=None)`，图像形状为 `[B, C, H, W]`。
输入数值应为浮点型；计算 PSNR、SSIM 时，`max_value`、`data_range` 必须与图像范围一致。

没有 mask 时，直接按标准公式计算整幅图像。有 mask 时，`mask > 0` 表示有效位置，
支持 `[B, H, W]`、`[B, 1, H, W]` 和 `[B, C, H, W]`。
不会根据图像中的零值自动生成 mask。训练验证和推理读取 batch 中可选的 `mask`；
两阶段推理还支持 `mask_stage_1`、`mask_stage_2`，各 mask 的尺寸必须对应阶段图像。

RMSE、MAE、PSNR 在全部有效波段像素上汇总；`is_reduce_channel=False` 返回各波段结果。
CC、UIQI、ERGAS 先计算每张图像、每个波段的统计量，再按参考实现的顺序汇总。
CC、UIQI 使用中心化的双精度统计，避免低方差区域的数值消减。
ERGAS 的 `ratio` 是乘法系数，当前配置使用 `1/16`，分母使用 GT 均值。

SAM 默认输出弧度，可设置 `unit='degree'`。一个像素必须全部波段有效且双方光谱范数非零，
才参与 SAM；不在范数分母上直接加 epsilon。全零光谱的角度未定义。

带 mask 的 SSIM 使用有效像素的高斯权重重新归一化，保留已有规则：
窗口中心必须有效，且窗口的有效高斯权重超过 0.5。没有 mask 时使用标准高斯窗口 SSIM。
RMSE_BD 在 mask 有效区域内、距离无效区域不超过指定半径的位置计算；无 mask 时返回普通 RMSE。

没有有效样本的指标或波段返回 NaN，统计时跳过这些未定义值；全部未定义时汇总仍为 NaN。
零误差 PSNR 返回正无穷。常量光谱的 CC 未定义；相同常量图像的 UIQI 为 1。
ERGAS 对 GT 均值和误差均为零的波段记为零，GT 均值为零但存在误差时为正无穷。
mask 外的 NaN/Inf 不参与计算；有效区域内的数据仍应为有限数值。

## 离线计算

`tools.offline_metric_cal.offline_evaluation_report` 和批量版本均支持可选参数：
`mask_dir_path`、`normalize_scale`、`gt_layout`、`pred_layout`。
未指定 mask 目录时，查找 GT 目录旁的 `mask` 或 `Mask` 目录；不存在时按无 mask 计算。
影像和 mask 按编号匹配，支持 `_L_`、`_M_`、`_save_img_`、`_mask_` 命名标记。
编号不一致、重复或缺失时直接报错，不按排序位置猜测配对。

CIA/LGC 当前使用 `normalize_scale=10000.0`；ML 当前配置的数据范围为 `[0, 1]`，
需设置 `normalize_scale=1.0`。默认布局为 HWC，预测文件为 CHW 时显式指定 `pred_layout='CHW'`。
逐波段结果单独保留，全局 RMSE/PSNR 使用全部有效波段像素重新计算，不取逐波段指标的简单平均。
离线最终汇总为各影像指标的平均；在线 Tracker 汇总为各次 update 的平均，batch size 大于 1 时应注意这一区别。

```python
from src.metrics import RMSE, SSIM, ERGAS, CC, SAM
from tools.offline_metric_cal import offline_evaluation_report

summary = offline_evaluation_report(
    gt_dir_path='path/to/ML/Landsat_02',
    pred_dir_path='path/to/predictions',
    exp_name='ML',
    metric_list=[RMSE(), SSIM(data_range=1), ERGAS(ratio=1/16), CC(), SAM()],
    output_dir='path/to/reports',
    mask_dir_path='path/to/ML/mask',
    normalize_scale=1.0,
)
```

## 验证

```bash
python -B -m unittest discover -s tests -v
```

测试覆盖标准公式、mask 等价性和有效像素计数、空 mask、低方差 CC、低亮度 SAM、
掩膜外 NaN/Inf、离线配对与汇总、推理 mask 传递和 STARFM 分块 mask 对齐。
流程测试加载真实评估函数，并以模型和日志桩替代 GPU 推理与日志依赖；不依赖真实数据和 checkpoint。
