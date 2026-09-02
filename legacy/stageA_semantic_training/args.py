from __init__ import *
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='模型和训练参数')

    # 基本模型参数
    parser.add_argument('-e', '--default_encoder', type=str,
                        default='maxvit_base_tf_512.in21k_ft_in1k',
                        help='默认编码器')
    parser.add_argument('-a', '--arch', type=str, default='unet',
                        help='模型架构')
    parser.add_argument('-w', '--encoder_weights', type=str,
                        default='imagenet', help='编码器权重')
    parser.add_argument('-m', '--model_name', type=str, default=None,
                        help='模型名称')
    parser.add_argument('-nc', '--num_classes', type=int, default=1,
                        help='类别数量')
    parser.add_argument('-ic', '--in_channels', type=int, default=3,
                        help='输入通道数')

    # 数据与训练基本参数
    parser.add_argument('-r', '--root', type=str,
                        default='data',
                        help='数据根目录 (下面有 train/val/test)')
    parser.add_argument('-g', '--gpus', nargs='+', type=int, default=[0],
                        help='使用的 GPU')
    parser.add_argument('-wr', '--workers', type=int, default=4,
                        help='dataloader worker 数量')
    parser.add_argument('-rs', '--rand_seed', type=int, default=3407,
                        help='随机种子')
    parser.add_argument('-bs', '--batch_size', type=int, default=4,
                        help='批量大小')
    parser.add_argument('-l', '--lr', type=float, default=3e-5,
                        help='学习率')
    parser.add_argument('-ep', '--epochs', type=int, default=500,
                        help='训练轮数 (上限)')
    parser.add_argument('-wp', '--warmup_epochs', type=int, default=5,
                        help='预热轮数')
    parser.add_argument('-up', '--upsample', type=bool, default=False,
                        help='是否上采样图片到 512 再推理')
    parser.add_argument('-dr', '--drop', type=float, default=0.00,
                        help='dropout 比例')
    parser.add_argument('-en', '--exp_name', type=str, default='20251123StageA',
                        help='实验名称/标记')

    # 一般损失设置（非 FP 阶段的老代码可用）
    parser.add_argument('-lw', '--loss_weight', type=int, default=100,
                        help='损失权重 (旧代码使用)')
    parser.add_argument('-ls', '--loss', type=str, default='focal',
                        help='损失函数名称 (旧训练脚本使用)')

    # 传统负样本下采样（旧 trainer.py 用）
    parser.add_argument('-nd', '--neg_downsample', type=float, default=3.0,
                        help='负样本在旧版 WeightedRandomSampler 中的下采样倍数')

    # ========== 下面是 CV + FP 抑制专用的新增参数 ==========

    # K 折数量
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='Cross-Validation 折数 (默认 5)')
    parser.add_argument('--start_fold', type=int, default=1,
                    help='从第几折开始训练（用于中断后从某一折重新开始）')

    # 正负像素比例 -> pos_weight 上限 / 扫描限制
    parser.add_argument('--pos_weight_cap', type=float, default=30.0,
                        help='BCE pos_weight 的最大上限')
    parser.add_argument('--pos_weight_scan_limit', type=int, default=0,
                        help='估计 pos_weight 时最多扫描多少个样本 (0 表示使用全部)')

    # 是否使用 TverskyLoss 以及参数
    parser.add_argument('--use_tversky', type=bool, default=True,
                        help='是否在 BCE 之外再加 TverskyLoss')
    parser.add_argument('--tversky_alpha', type=float, default=0.3,
                        help='Tversky Loss 中的 alpha (FN 权重)')
    parser.add_argument('--tversky_beta', type=float, default=0.7,
                        help='Tversky Loss 中的 beta (FP 权重)')

    # 纯背景 patch 的 loss 放大系数
    parser.add_argument('--bg_loss_scale', type=float, default=2.5,
                        help='对纯背景样本 BCE loss 的放大系数')

    # CV 阶段负样本采样权重 (trainer_cv_fp 中使用)
    parser.add_argument('--neg_sample_weight', type=float, default=1.2,
                        help='CV 阶段 WeightedRandomSampler 中负样本的权重')

    # FP 抑制相关: 验证时对 Neg FP 的目标上限, 以及 trade-off 系数
    parser.add_argument('--target_neg_fp', type=float, default=0.15,
                        help='选择阈值时希望负样本 FP 不超过此值')
    parser.add_argument('--lambda_fp', type=float, default=1.0,
                        help='当无法满足 FP 约束时, score = IoU(pos) - lambda_fp * NegFP 中的系数')

    # AMP / EMA / early stopping
    parser.add_argument('--amp', type=bool, default=True,
                        help='是否使用混合精度训练')
    parser.add_argument('--ema', type=bool, default=True,
                        help='是否使用模型 EMA')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA 衰减系数')
    parser.add_argument('--patience', type=int, default=20,
                        help='早停耐心轮数 (验证指标长期不提升则停止)')

    # 评估相关
    parser.add_argument('--eval_split_mask', action='store_true',
                        help='验证时是否分开统计正负样本 IoU')
        # topology / clDice
    parser.add_argument('--use_cldice', type=bool, default=True)
    parser.add_argument('--cldice_weight', type=float, default=0.5)
    parser.add_argument('--cldice_iters', type=int, default=12)

    # centerline head
    parser.add_argument('--use_centerline_head', type=bool, default=True)
    parser.add_argument('--centerline_weight', type=float, default=0.3)

    # hard negative mining
    parser.add_argument('--hardneg', type=bool, default=True)
    parser.add_argument('--hardneg_start_epoch', type=int, default=3)
    parser.add_argument('--hardneg_gamma', type=float, default=5.0)
    parser.add_argument('--hardneg_max_weight', type=float, default=5.0)
    parser.add_argument('--hardneg_score_limit', type=int, default=0)  # 0=全扫
    # ===== schedule ratios (curriculum within StageA) =====
    parser.add_argument('--cldice_start_ratio', type=float, default=0.2,
                        help='训练进度超过该比例后启用 clDice，并从0线性ramp到 cldice_weight')
    parser.add_argument('--cen_start_ratio', type=float, default=0.2,
                        help='训练进度超过该比例后启用 centerline loss，并从0线性ramp到 centerline_weight')
    parser.add_argument('--hardneg_start_ratio', type=float, default=0.35,
                        help='训练进度超过该比例后启用 hard negative reweight')

    args = parser.parse_args()

    # 如果未通过命令行指定 model_name，则按原始规则构建
    if args.model_name is None:
        args.model_name = "{}_{}_{}_{}".format(
            args.arch, args.default_encoder, args.loss, args.exp_name
        )

    # 模型/损失输出路径
    args.MODEL_PATH = 'ckpts/{}'.format(args.model_name)
    os.makedirs(args.MODEL_PATH, exist_ok=True)

    args.LOSS_PATH = 'comeout/loss'
    os.makedirs(args.LOSS_PATH, exist_ok=True)

    args.LOSS_NPY_PATH = 'comeout/loss_npy/{}'.format(args.model_name)
    os.makedirs(args.LOSS_NPY_PATH, exist_ok=True)

    return args

args = parse_args()
