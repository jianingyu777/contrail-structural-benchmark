# # 1117banben
# # from __future__ import annotations
# # import os, cv2, gc, json, random, glob, warnings
# # from pathlib import Path
# # from typing import List, Tuple, Dict

# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# # from tqdm import tqdm
# # import segmentation_models_pytorch as smp

# # try:
# #     from sklearn.model_selection import StratifiedKFold
# #     _SKLEARN = True
# # except Exception:
# #     StratifiedKFold = None
# #     _SKLEARN = False

# # from __init__ import *
# # from args import *
# # from data_aug import transform_train, transform_val
# # from models.seg_model import *
# # from models.timm_model import *
# # from lrscheduler import *

# # warnings.filterwarnings("ignore")

# # IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


# # def read_mask_binary_uint8(p: str) -> np.ndarray:
# #     m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
# #     if m is None:
# #         raise RuntimeError(f"Error loading mask: {p}")
# #     if m.max() > 1:
# #         _, m = cv2.threshold(m, 127, 1, cv2.THRESH_BINARY)
# #     return m.astype(np.uint8)


# # def scan_pairs(base: str) -> List[Tuple[str, str]]:
# #     img_dir = os.path.join(base, 'image')
# #     lbl_dir = os.path.join(base, 'label')
# #     if not (os.path.isdir(img_dir) and os.path.isdir(lbl_dir)):
# #         return []
# #     img_paths = [p for p in glob.glob(os.path.join(img_dir, '*'))
# #                  if Path(p).suffix.lower() in IMG_EXTS]
# #     lbl_paths = [p for p in glob.glob(os.path.join(lbl_dir, '*'))
# #                  if Path(p).suffix.lower() in IMG_EXTS]
# #     imap = {Path(p).stem: p for p in img_paths}
# #     lmap = {Path(p).stem: p for p in lbl_paths}
# #     stems = sorted(set(imap) & set(lmap))
# #     return [(imap[s], lmap[s]) for s in stems]


# # def estimate_pos_weight_from_items(items: List[Tuple[str,str]],
# #                                    cap: float=20.0,
# #                                    limit: int=0) -> float:
# #     total_pos = 0
# #     total_pixels = 0
# #     iterable = items if limit <= 0 else items[:limit]
# #     for _, mpath in iterable:
# #         m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
# #         if m is None:
# #             continue
# #         pos = (m > 0).sum()
# #         total_pos += int(pos)
# #         total_pixels += int(m.size)
# #     total_neg = max(1, total_pixels - total_pos)
# #     total_pos = max(1, total_pos)
# #     pw = float(total_neg / total_pos)
# #     pw = float(min(cap, max(1.0, pw)))
# #     return pw


# # def build_strata(items: List[Tuple[str,str]]) -> Tuple[List[int], Dict[int,str]]:
# #     strata_names = []
# #     bins = getattr(args, "pos_bins", [0.0, 0.01, 0.05, 1.01])
# #     for _, mpath in items:
# #         m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
# #         if m is None:
# #             raise RuntimeError(f"Error loading mask: {mpath}")
# #         if m.sum() == 0:
# #             strata_names.append('neg')
# #         else:
# #             r = (m > 0).sum() / m.size
# #             if r <= bins[1]:
# #                 strata_names.append('pos_small')
# #             elif r <= bins[2]:
# #                 strata_names.append('pos_mid')
# #             else:
# #                 strata_names.append('pos_large')
# #     uniq = {name: i for i, name in enumerate(sorted(set(strata_names)))}
# #     strata_ids = [uniq[n] for n in strata_names]
# #     id2name = {i: n for n, i in uniq.items()}
# #     return strata_ids, id2name


# # class CVPairDataset(Dataset):
# #     def __init__(self, items: List[Tuple[str,str]], transform):
# #         self.items = items
# #         self.t = transform

# #     def __len__(self):
# #         return len(self.items)

# #     def __getitem__(self, idx):
# #         ip, mp = self.items[idx]
# #         img = cv2.imread(ip, cv2.IMREAD_COLOR)
# #         if img is None:
# #             raise RuntimeError(f'Error loading {ip}')
# #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
# #         m = read_mask_binary_uint8(mp)
# #         out = self.t(image=img, mask=m)
# #         image = out['image']
# #         mask = (out['mask'].float() > 0.5).float().unsqueeze(0)
# #         return image, mask


# # class ModelEMA:
# #     def __init__(self, model, decay=0.999):
# #         from copy import deepcopy
# #         self.ema = deepcopy(model).eval()
# #         for p in self.ema.parameters():
# #             p.requires_grad_(False)
# #         self.decay = decay

# #     @torch.no_grad()
# #     def update(self, model):
# #         d = self.decay
# #         msd = model.state_dict()
# #         for k, v in self.ema.state_dict().items():
# #             if k in msd:
# #                 v.copy_(v * d + msd[k] * (1.0 - d))


# # def build_sampler_for_subset(items_subset: List[Tuple[str,str]],
# #                              neg_sample_weight: float = 0.6):
# #     pos_idx, neg_idx = [], []
# #     for i, (_, mpath) in enumerate(items_subset):
# #         m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
# #         if m is None:
# #             raise RuntimeError(f"Error loading mask: {mpath}")
# #         if m.sum() > 0:
# #             pos_idx.append(i)
# #         else:
# #             neg_idx.append(i)
# #     weights = [0.0] * len(items_subset)
# #     for i in pos_idx:
# #         weights[i] = 1.0
# #     for i in neg_idx:
# #         weights[i] = max(1e-6, float(neg_sample_weight))
# #     return WeightedRandomSampler(weights=weights,
# #                                  num_samples=len(items_subset),
# #                                  replacement=True)


# # def eval_on_loader(model,
# #                    loader,
# #                    device,
# #                    compute_loss,
# #                    thr_list,
# #                    use_amp=False,
# #                    upsample=False,
# #                    target_neg_fp=0.2,
# #                    lambda_fp=0.5):
# #     model.eval()
# #     losses = []
# #     pos_inter = {thr: 0.0 for thr in thr_list}
# #     pos_union = {thr: 0.0 for thr in thr_list}
# #     neg_ok = {thr: 0.0 for thr in thr_list}
# #     neg_cnt = 0.0
# #     amp = bool(use_amp)

# #     with torch.no_grad():
# #         for X, y in loader:
# #             X = X.to(device, non_blocking=True)
# #             y = y.to(device, non_blocking=True)

# #             with torch.cuda.amp.autocast(enabled=amp):
# #                 out = model(X if not upsample else F.interpolate(
# #                     X, scale_factor=2, mode='bicubic', align_corners=False))
# #                 loss = compute_loss(
# #                     out,
# #                     y if not upsample else F.interpolate(
# #                         y, scale_factor=2, mode='nearest')
# #                 )
# #             losses.append(loss.item())

# #             prob = torch.sigmoid(out)
# #             if upsample:
# #                 prob = F.interpolate(prob, scale_factor=0.5, mode='nearest')

# #             for thr in thr_list:
# #                 pb = (prob > thr).float()
# #                 for pbi, gti in zip(pb, y):
# #                     if gti.sum() == 0:
# #                         neg_cnt += 1.0
# #                         neg_ok[thr] += 1.0 if pbi.sum() == 0 else 0.0
# #                     else:
# #                         inter = torch.logical_and(
# #                             pbi > 0.5, gti > 0.5).sum().item()
# #                         union = torch.logical_or(
# #                             pbi > 0.5, gti > 0.5).sum().item()
# #                         pos_inter[thr] += inter
# #                         pos_union[thr] += max(1.0, union)

# #     avg_loss = float(np.mean(losses)) if losses else 0.0

# #     # 先在满足 Neg FP 约束下最大化 IoU(pos)
# #     best_thr, best_pos_iou, best_neg_fp = thr_list[0], 0.0, 1.0
# #     chosen = False
# #     for thr in thr_list:
# #         pos_iou = (pos_inter[thr] / pos_union[thr]) if pos_union[thr] > 0 else 0.0
# #         neg_fp = 1.0 - (neg_ok[thr] / neg_cnt) if neg_cnt > 0 else 0.0
# #         if neg_fp <= target_neg_fp:
# #             if (not chosen) or (pos_iou > best_pos_iou):
# #                 best_thr, best_pos_iou, best_neg_fp = thr, pos_iou, neg_fp
# #                 chosen = True

# #     # 若没有阈值满足 FP 约束，则用带惩罚的目标选一个
# #     if not chosen:
# #         best_score = -1e9
# #         for thr in thr_list:
# #             pos_iou = (pos_inter[thr] / pos_union[thr]) if pos_union[thr] > 0 else 0.0
# #             neg_fp = 1.0 - (neg_ok[thr] / neg_cnt) if neg_cnt > 0 else 0.0
# #             score = pos_iou - lambda_fp * neg_fp
# #             if score > best_score:
# #                 best_score = score
# #                 best_thr, best_pos_iou, best_neg_fp = thr, pos_iou, neg_fp

# #     # mIoU(all) 估计
# #     mIoU_all = 0.0
# #     if pos_union[best_thr] + neg_cnt > 0:
# #         mIoU_all = (pos_inter[best_thr] + neg_ok[best_thr]) / (pos_union[best_thr] + neg_cnt)

# #     return avg_loss, best_thr, best_pos_iou, best_neg_fp, mIoU_all


# # def trainer_cv_fp():
# #     print("========== 5-Fold CV (FP-suppression) ==========")
# #     np.random.seed(args.rand_seed)
# #     random.seed(args.rand_seed)
# #     torch.manual_seed(args.rand_seed)
# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed_all(args.rand_seed)
# #     torch.backends.cudnn.benchmark = False
# #     torch.backends.cudnn.deterministic = True

# #     # 收集样本池（train + val 目录下的 image/label）
# #     pool_items = []
# #     for sub in ['train', 'val']:
# #         base = os.path.join(args.root, sub)
# #         if os.path.isdir(base):
# #             pool_items += scan_pairs(base)
# #     if len(pool_items) == 0:
# #         raise RuntimeError("No data found under root/train or root/val (image/label).")

# #     strata_ids, id2name = build_strata(pool_items)
# #     print("Strata:", {i: n for i, n in id2name.items()})

# #     K = int(getattr(args, "cv_folds", 5))
# #     thr_list = [round(x, 2) for x in np.linspace(0.2, 0.95, 16)]
# #     target_neg_fp = float(getattr(args, "target_neg_fp", 0.2))
# #     lambda_fp = float(getattr(args, "lambda_fp", 0.5))

# #     if _SKLEARN:
# #         from sklearn.model_selection import StratifiedKFold
# #         skf = StratifiedKFold(n_splits=K,
# #                               shuffle=True,
# #                               random_state=args.rand_seed)
# #         splits = skf.split(np.arange(len(pool_items)),
# #                            np.array(strata_ids))
# #     else:
# #         by_sid = {}
# #         for i, sid in enumerate(strata_ids):
# #             by_sid.setdefault(sid, []).append(i)
# #         for s in by_sid.values():
# #             random.Random(args.rand_seed).shuffle(s)

# #         def gen():
# #             for f in range(K):
# #                 val_idx = []
# #                 for sid, idxs in by_sid.items():
# #                     val_idx += idxs[f::K]
# #                 train_idx = sorted(
# #                     list(set(range(len(pool_items))) - set(val_idx)))
# #                 yield train_idx, val_idx
# #         splits = gen()

# #     per_fold_iou = []
# #     start_fold = int(getattr(args, "start_fold", 1))
# #     for fold, (tr_idx, va_idx) in enumerate(splits, 1):
# #         if fold < start_fold:
# #             print(f"\n----- Fold {fold}/{K} skipped (start_fold={start_fold}) -----")
# #             # 如果已经有对应的 final.pth，可以记录一下占位的 IoU，或者简单 append 0
# #             per_fold_iou.append(0.0)
# #             continue
# #         print(f"\n----- Fold {fold}/{K} -----")
# #         items_train = [pool_items[i] for i in tr_idx]
# #         items_val = [pool_items[i] for i in va_idx]

# #         ds_train = CVPairDataset(items_train, transform_train)
# #         ds_val = CVPairDataset(items_val, transform_val)

# #         sampler = build_sampler_for_subset(
# #             items_train,
# #             neg_sample_weight=float(getattr(args, "neg_sample_weight", 0.6))
# #         )
# #         dl_train = DataLoader(
# #             ds_train,
# #             batch_size=args.batch_size,
# #             sampler=sampler,
# #             num_workers=args.workers,
# #             pin_memory=True,
# #             drop_last=True
# #         )
# #         dl_val = DataLoader(
# #             ds_val,
# #             batch_size=args.batch_size,
# #             shuffle=False,
# #             num_workers=args.workers,
# #             pin_memory=True,
# #             drop_last=False
# #         )

# #         model = Model()
# #         if len(args.gpus) == 0:
# #             device = torch.device('cpu')
# #         elif len(args.gpus) == 1:
# #             torch.cuda.set_device(args.gpus[0])
# #             model.cuda()
# #             device = torch.device('cuda')
# #         else:
# #             gpus = ','.join(str(i) for i in args.gpus)
# #             os.environ["CUDA_VISIBLE_DEVICES"] = gpus
# #             model.cuda()
# #             model = torch.nn.DataParallel(
# #                 model, device_ids=list(range(len(args.gpus))))
# #             device = torch.device('cuda')

# #         # cap = float(getattr(args, "pos_weight_cap", 30.0))
# #         # scan_limit = int(getattr(args, "pos_weight_scan_limit", 0))
# #         # est_pw = estimate_pos_weight_from_items(
# #         #     items_train, cap=cap, limit=scan_limit)
# #         # bce = nn.BCEWithLogitsLoss(
# #         #     pos_weight=torch.tensor([est_pw], device=device))
# #         # 替换原来那段自动估计的逻辑：
# #         pos_weight_value = 5.0  # 可以先试 5，再看曲线微调
# #         bce = nn.BCEWithLogitsLoss(
# #             pos_weight=torch.tensor([pos_weight_value], device=device)
# #         )

# #         use_tversky = bool(getattr(args, "use_tversky", True))
# #         if use_tversky:
# #             alpha = float(getattr(args, "tversky_alpha", 0.4))
# #             beta = float(getattr(args, "tversky_beta", 0.6))
# #             aux_loss = smp.losses.TverskyLoss(
# #                 mode='binary', from_logits=True, alpha=alpha, beta=beta)
# #         else:
# #             aux_loss = smp.losses.DiceLoss(
# #                 mode='binary', from_logits=True)

# #         bg_loss_scale = float(getattr(args, "bg_loss_scale", 1.8))

# #         def compute_loss(pred, tgt):
# #             b = bce(pred, tgt)
# #             if tgt.sum() == 0:
# #                 return bg_loss_scale * b
# #             else:
# #                 return b + aux_loss(pred, tgt)

# #         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# #         scheduler = get_cosine_schedule_with_warmup(
# #             optimizer,
# #             num_warmup_steps=args.warmup_epochs,
# #             num_training_steps=args.epochs
# #         )
# #         use_amp = bool(getattr(args, 'amp', True)) and torch.cuda.is_available()
# #         scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
# #         ema = ModelEMA(model, decay=float(getattr(args, 'ema_decay', 0.999))) \
# #             if bool(getattr(args, 'ema', True)) else None

# #         patience = int(getattr(args, 'patience', 15))
# #         patience_cnt = 0
# #         best_iou, best_thr, best_val_loss = -1.0, 0.5, 1e9
# #         # ⭐ 新增：同时考虑 IoU(pos) 和 Neg FP 的综合 score
# #         best_score = -1e9
# #         best_neg_fp = 1.0

# #         best_iou_path = os.path.join(
# #             args.MODEL_PATH,
# #             f"{args.model_name}_fp_fold{fold}_best_iou.pth"
# #         )
# #         best_loss_path = os.path.join(
# #             args.MODEL_PATH,
# #             f"{args.model_name}_fp_fold{fold}_best_loss.pth"
# #         )
# #         final_path = os.path.join(
# #             args.MODEL_PATH,
# #             f"{args.model_name}_fp_fold{fold}_final.pth"
# #         )

# #         for epoch in range(1, args.epochs + 1):
# #             model.train()
# #             losses = []
# #             for X, y in tqdm(dl_train):
# #                 X = X.to(device, non_blocking=True)
# #                 y = y.to(device, non_blocking=True)
# #                 optimizer.zero_grad(set_to_none=True)
# #                 with torch.cuda.amp.autocast(enabled=use_amp):
# #                     out = model(X if not args.upsample else F.interpolate(
# #                         X, scale_factor=2, mode='bicubic', align_corners=False))
# #                     loss = compute_loss(
# #                         out,
# #                         y if not args.upsample else F.interpolate(
# #                             y, scale_factor=2, mode='nearest')
# #                     )
# #                 scaler.scale(loss).backward()
# #                 scaler.step(optimizer)
# #                 scaler.update()
# #                 if ema is not None:
# #                     ema.update(model)
# #                 losses.append(loss.item())
# #             scheduler.step()
# #             print(f"[Fold {fold}] [Train {epoch:03d}/{args.epochs:03d}] loss={float(np.mean(losses)):.5f}")

# #             # ===== 验证阶段 =====
# #             eval_model = ema.ema if ema is not None else model
# #             val_loss, thr, pos_iou, neg_fp, mIoU_all = eval_on_loader(
# #                 eval_model,
# #                 dl_val,
# #                 device,
# #                 compute_loss,
# #                 thr_list,
# #                 use_amp=use_amp,
# #                 upsample=args.upsample,
# #                 target_neg_fp=target_neg_fp,
# #                 lambda_fp=lambda_fp
# #             )
# #             print(
# #                 f"[Fold {fold}] [Valid {epoch:03d}/{args.epochs:03d}] "
# #                 f"loss={val_loss:.5f}, IoU(pos)={pos_iou:.4f} @thr={thr:.2f}, "
# #                 f"Neg FP={neg_fp:.4f}, mIoU(all)={mIoU_all:.4f}"
# #             )

# #             # 保存“最小验证 loss”模型（作为参考）
# #             if val_loss < best_val_loss:
# #                 best_val_loss = val_loss
# #                 torch.save(eval_model.state_dict(), best_loss_path)

# #             # ⭐ 核心修改：用 score = IoU(pos) - lambda_fp * NegFP 做综合评价
# #             score = pos_iou - lambda_fp * neg_fp

# #             # 如果你想“硬约束 Neg FP <= target_neg_fp”，可以用下面这个替代:
# #             # if (neg_fp <= target_neg_fp) and (score > best_score):
# #             #     ...

# #             if score > best_score:
# #                 best_score = score
# #                 best_iou = pos_iou
# #                 best_thr = thr
# #                 best_neg_fp = neg_fp
# #                 torch.save(eval_model.state_dict(), best_iou_path)
# #                 patience_cnt = 0
# #             else:
# #                 patience_cnt += 1

# #             if patience_cnt >= patience:
# #                 print(f"[Fold {fold}] Early stopping at epoch {epoch}.")
# #                 break

# #         # 保存最后一轮的模型
# #         torch.save((ema.ema if ema is not None else model).state_dict(), final_path)
# #         print(f"[Fold {fold}] best_IoU={best_iou:.4f} @thr={best_thr:.2f}, best_NegFP={best_neg_fp:.4f}")

# #         per_fold_iou.append(best_iou)
# #         del ds_train, ds_val, dl_train, dl_val
# #         gc.collect()

# #     os.makedirs(args.MODEL_PATH, exist_ok=True)
# #     with open(os.path.join(args.MODEL_PATH,
# #                            f"{args.model_name}_cv_fp_summary.json"), "w") as f:
# #         json.dump({"per_fold_iou": per_fold_iou}, f, indent=2)

# #     print("\n========== CV(FP) Finished ==========")
# #     print("Per-fold IoU(pos):", [round(x, 4) for x in per_fold_iou])
# #     return per_fold_iou
# from __future__ import annotations
# import os, cv2, gc, json, random, glob, warnings
# from pathlib import Path
# from typing import List, Tuple, Dict, Optional

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# from tqdm import tqdm
# import segmentation_models_pytorch as smp

# try:
#     from sklearn.model_selection import StratifiedKFold
#     _SKLEARN = True
# except Exception:
#     StratifiedKFold = None
#     _SKLEARN = False

# from __init__ import *
# from args import *
# from data_aug import transform_train, transform_val
# from models.timm_model import Model as BaseModel  # 你的原始 Model
# from lrscheduler import *

# warnings.filterwarnings("ignore")
# IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


# # =========================
# # Utils
# # =========================
# def read_mask_binary_uint8(p: str) -> np.ndarray:
#     m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
#     if m is None:
#         raise RuntimeError(f"Error loading mask: {p}")
#     if m.max() > 1:
#         _, m = cv2.threshold(m, 127, 1, cv2.THRESH_BINARY)
#     return m.astype(np.uint8)

# def scan_pairs(base: str) -> List[Tuple[str, str]]:
#     img_dir = os.path.join(base, 'image')
#     lbl_dir = os.path.join(base, 'label')
#     if not (os.path.isdir(img_dir) and os.path.isdir(lbl_dir)):
#         return []
#     img_paths = [p for p in glob.glob(os.path.join(img_dir, '*'))
#                  if Path(p).suffix.lower() in IMG_EXTS]
#     lbl_paths = [p for p in glob.glob(os.path.join(lbl_dir, '*'))
#                  if Path(p).suffix.lower() in IMG_EXTS]
#     imap = {Path(p).stem: p for p in img_paths}
#     lmap = {Path(p).stem: p for p in lbl_paths}
#     stems = sorted(set(imap) & set(lmap))
#     return [(imap[s], lmap[s]) for s in stems]

# def build_strata(items: List[Tuple[str,str]]) -> Tuple[List[int], Dict[int,str]]:
#     strata_names = []
#     bins = getattr(args, "pos_bins", [0.0, 0.01, 0.05, 1.01])
#     for _, mpath in items:
#         m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
#         if m is None:
#             raise RuntimeError(f"Error loading mask: {mpath}")
#         if m.sum() == 0:
#             strata_names.append('neg')
#         else:
#             r = (m > 0).sum() / m.size
#             if r <= bins[1]: strata_names.append('pos_small')
#             elif r <= bins[2]: strata_names.append('pos_mid')
#             else: strata_names.append('pos_large')
#     uniq = {name:i for i,name in enumerate(sorted(set(strata_names)))}
#     strata_ids = [uniq[n] for n in strata_names]
#     id2name = {i:n for n,i in uniq.items()}
#     return strata_ids, id2name


# # =========================
# # Centerline target (CPU skeleton)
# # =========================
# def skeletonize_binary(mask_np: np.ndarray) -> np.ndarray:
#     """
#     mask_np: uint8 0/1
#     return: uint8 0/1 skeleton (thin centerline)
#     """
#     mask = (mask_np > 0).astype(np.uint8)
#     if mask.sum() == 0:
#         return mask
#     skel = np.zeros_like(mask)
#     element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
#     img = mask.copy()
#     while True:
#         eroded = cv2.erode(img, element)
#         opened = cv2.dilate(eroded, element)
#         temp = cv2.subtract(img, opened)
#         skel = cv2.bitwise_or(skel, temp)
#         img = eroded.copy()
#         if cv2.countNonZero(img) == 0:
#             break
#     skel = (skel > 0).astype(np.uint8)
#     return skel


# # =========================
# # clDice (soft skeleton, differentiable)
# # =========================
# def soft_erode(img):
#     p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
#     p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
#     return torch.min(p1, p2)

# def soft_dilate(img):
#     return F.max_pool2d(img, 3, 1, 1)

# def soft_open(img):
#     return soft_dilate(soft_erode(img))

# def soft_skel(img, iters=10):
#     skel = F.relu(img - soft_open(img))
#     for _ in range(iters):
#         img = soft_erode(img)
#         opened = soft_open(img)
#         delta = F.relu(img - opened)
#         skel = skel + F.relu(delta - skel * delta)
#     return skel

# def cldice_loss_from_logits(pred_logits, gt, iters=10, smooth=1.0):
#     """
#     pred_logits: (B,1,H,W) logits
#     gt         : (B,1,H,W) float 0/1
#     """
#     pred = torch.sigmoid(pred_logits)
#     skel_pred = soft_skel(pred, iters)
#     skel_gt   = soft_skel(gt, iters)

#     dims = (1,2,3)
#     tprec = (skel_pred * gt).sum(dims) / (skel_pred.sum(dims) + smooth)
#     tsens = (skel_gt * pred).sum(dims) / (skel_gt.sum(dims) + smooth)

#     cldice = (2 * tprec * tsens + smooth) / (tprec + tsens + smooth)
#     return 1.0 - cldice.mean()


# # =========================
# # Dataset (return centerline if needed)
# # =========================
# class CVPairDataset(Dataset):
#     def __init__(self, items: List[Tuple[str,str]], transform, with_centerline: bool=False):
#         self.items = items
#         self.t = transform
#         self.with_centerline = with_centerline

#     def __len__(self): return len(self.items)

#     def __getitem__(self, idx):
#         ip, mp = self.items[idx]
#         img = cv2.imread(ip, cv2.IMREAD_COLOR)
#         if img is None: raise RuntimeError(f'Error loading {ip}')
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
#         m = read_mask_binary_uint8(mp)

#         out = self.t(image=img, mask=m)
#         image = out['image']
#         mask_t = out['mask']
#         if not isinstance(mask_t, torch.Tensor):
#             mask_t = torch.from_numpy(mask_t)
#         mask = (mask_t.float() > 0.5).float().unsqueeze(0)

#         if not self.with_centerline:
#             return image, mask

#         # centerline target
#         m_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)
#         cen_np = skeletonize_binary(m_np)
#         cen = torch.from_numpy(cen_np).float().unsqueeze(0)
#         return image, mask, cen


# # =========================
# # EMA
# # =========================
# class ModelEMA:
#     def __init__(self, model, decay=0.999):
#         from copy import deepcopy
#         self.ema = deepcopy(model).eval()
#         for p in self.ema.parameters(): p.requires_grad_(False)
#         self.decay = decay

#     @torch.no_grad()
#     def update(self, model):
#         d = self.decay
#         msd = model.state_dict()
#         for k, v in self.ema.state_dict().items():
#             if k in msd:
#                 v.copy_(v*d + msd[k]*(1.0-d))


# # =========================
# # Multi-head model (seg + centerline)
# # =========================
# class CenterlineModel(BaseModel):
#     """
#     在你原 BaseModel 基础上加一个 centerline head，forward 返回 (seg_logits, center_logits)
#     """
#     def __init__(self, pretrained=True, decoder_channels=(384,192,96,64), out_indices=(0,1,2,3)):
#         super().__init__(pretrained=pretrained, decoder_channels=decoder_channels, out_indices=out_indices)
#         self.centerline_head = smp.base.SegmentationHead(
#             in_channels=decoder_channels[-1],
#             out_channels=1, activation=None, kernel_size=3,
#         )

#     def forward(self, x):
#         features = self.encoder(x)
#         decoder_output = self.decoder(features)
#         seg_logits = self.segmentation_head(decoder_output)
#         cen_logits = self.centerline_head(decoder_output)
#         return seg_logits, cen_logits


# # =========================
# # Sampler (pos/neg base)
# # =========================
# def build_sampler_for_subset(items_subset: List[Tuple[str,str]], neg_sample_weight: float=0.6):
#     pos_idx, neg_idx = [], []
#     for i, (_, mpath) in enumerate(items_subset):
#         m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
#         if m is None: raise RuntimeError(f"Error loading mask: {mpath}")
#         if m.sum() > 0: pos_idx.append(i)
#         else: neg_idx.append(i)

#     weights = [0.0]*len(items_subset)
#     for i in pos_idx: weights[i] = 1.0
#     for i in neg_idx: weights[i] = max(1e-6, float(neg_sample_weight))
#     return weights


# # =========================
# # Hard-neg mining (epoch-wise reweight)
# # =========================
# @torch.no_grad()
# def update_hardneg_weights(model,
#                            items_train: List[Tuple[str,str]],
#                            device,
#                            base_neg_weight: float,
#                            hardneg_gamma: float,
#                            max_weight: float,
#                            score_limit: int=0):
#     """
#     对 neg patch 根据当前模型的 FP 倾向重新加权
#     weight_neg = base_neg_weight * (1 + hardneg_gamma * fp_score)
#     fp_score 用 pred prob 的 max 值表征
#     """
#     # 只做 val transform（无随机增强）以稳定评分
#     ds_score = CVPairDataset(items_train, transform_val, with_centerline=False)
#     dl_score = DataLoader(ds_score, batch_size=args.batch_size, shuffle=False,
#                           num_workers=args.workers, pin_memory=True)

#     weights = []
#     neg_scores = []
#     neg_indices = []

#     model.eval()
#     for bi, batch in enumerate(dl_score):
#         X, y = batch
#         X = X.to(device, non_blocking=True)
#         y = y.to(device, non_blocking=True)

#         out = model(X)
#         if isinstance(out, (tuple, list)):
#             seg_logits = out[0]
#         else:
#             seg_logits = out

#         prob = torch.sigmoid(seg_logits)

#         for i in range(prob.shape[0]):
#             yi = y[i]
#             if yi.sum() == 0:
#                 fp_score = float(prob[i].max().item())  # 用 max 更敏感
#                 neg_scores.append(fp_score)
#                 neg_indices.append(bi*args.batch_size + i)

#         if score_limit > 0 and len(neg_scores) >= score_limit:
#             break

#     # 构造新权重
#     base_weights = build_sampler_for_subset(items_train, base_neg_weight)
#     for idx, fp_score in zip(neg_indices, neg_scores):
#         w = base_neg_weight * (1.0 + hardneg_gamma * fp_score)
#         w = float(min(max_weight, max(1e-6, w)))
#         base_weights[idx] = w

#     return base_weights


# # =========================
# # Eval
# # =========================
# def eval_on_loader(model, loader, device, compute_loss, thr_list,
#                    use_amp=False, upsample=False,
#                    target_neg_fp=0.2, lambda_fp=0.5):
#     model.eval()
#     losses = []
#     pos_inter = {thr:0.0 for thr in thr_list}
#     pos_union = {thr:0.0 for thr in thr_list}
#     neg_ok   = {thr:0.0 for thr in thr_list}
#     neg_cnt  = 0.0
#     amp = bool(use_amp)

#     with torch.no_grad():
#         for batch in loader:
#             if len(batch) == 2:
#                 X, y = batch
#                 c = None
#             else:
#                 X, y, c = batch

#             X = X.to(device, non_blocking=True)
#             y = y.to(device, non_blocking=True)
#             if c is not None:
#                 c = c.to(device, non_blocking=True)

#             with torch.cuda.amp.autocast(enabled=amp):
#                 out = model(X if not upsample else F.interpolate(X, scale_factor=2, mode='bicubic', align_corners=False))
#                 loss = compute_loss(out, y if not upsample else F.interpolate(y, scale_factor=2, mode='nearest'),
#                                     c if c is None else (c if not upsample else F.interpolate(c, scale_factor=2, mode='nearest')))
#             losses.append(loss.item())

#             # seg only for metric
#             seg_logits = out[0] if isinstance(out, (tuple, list)) else out
#             prob = torch.sigmoid(seg_logits)
#             if upsample:
#                 prob = F.interpolate(prob, scale_factor=0.5, mode='nearest')

#             for thr in thr_list:
#                 pb = (prob > thr).float()
#                 for pbi, gti in zip(pb, y):
#                     if gti.sum() == 0:
#                         neg_cnt += 1.0
#                         neg_ok[thr] += 1.0 if pbi.sum() == 0 else 0.0
#                     else:
#                         inter = torch.logical_and(pbi>0.5, gti>0.5).sum().item()
#                         union = torch.logical_or (pbi>0.5, gti>0.5).sum().item()
#                         pos_inter[thr] += inter
#                         pos_union[thr] += max(1.0, union)

#     avg_loss = float(np.mean(losses)) if losses else 0.0

#     # 先满足 Neg FP 约束下最大化 IoU(pos)
#     best_thr, best_pos_iou, best_neg_fp = thr_list[0], 0.0, 1.0
#     chosen = False
#     for thr in thr_list:
#         pos_iou = (pos_inter[thr] / pos_union[thr]) if pos_union[thr] > 0 else 0.0
#         neg_fp  = 1.0 - (neg_ok[thr] / neg_cnt) if neg_cnt > 0 else 0.0
#         if neg_fp <= target_neg_fp:
#             if (not chosen) or (pos_iou > best_pos_iou):
#                 best_thr, best_pos_iou, best_neg_fp = thr, pos_iou, neg_fp
#                 chosen = True

#     # 若没有阈值满足 FP 约束，则用带惩罚的目标选一个
#     if not chosen:
#         best_score = -1e9
#         for thr in thr_list:
#             pos_iou = (pos_inter[thr] / pos_union[thr]) if pos_union[thr] > 0 else 0.0
#             neg_fp  = 1.0 - (neg_ok[thr] / neg_cnt) if neg_cnt > 0 else 0.0
#             score = pos_iou - lambda_fp * neg_fp
#             if score > best_score:
#                 best_score = score
#                 best_thr, best_pos_iou, best_neg_fp = thr, pos_iou, neg_fp

#     mIoU_all = 0.0
#     if pos_union[best_thr] + neg_cnt > 0:
#         mIoU_all = (pos_inter[best_thr] + neg_ok[best_thr]) / (pos_union[best_thr] + neg_cnt)

#     return avg_loss, best_thr, best_pos_iou, best_neg_fp, mIoU_all


# # =========================
# # Trainer
# # =========================
# def trainer_cv_fp():
#     print("========== 5-Fold CV (FP-suppression + clDice + centerline + hardneg) ==========")
#     np.random.seed(args.rand_seed); random.seed(args.rand_seed); torch.manual_seed(args.rand_seed)
#     if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.rand_seed)
#     torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

#     # 收集样本池（train + val）
#     pool_items = []
#     for sub in ['train', 'val']:
#         base = os.path.join(args.root, sub)
#         if os.path.isdir(base): pool_items += scan_pairs(base)
#     if len(pool_items) == 0:
#         raise RuntimeError("No data found under root/train or root/val (image/label).")

#     strata_ids, id2name = build_strata(pool_items)
#     print("Strata:", {i:n for i,n in id2name.items()})

#     K = int(getattr(args, "cv_folds", 5))
#     thr_list = [round(x,2) for x in np.linspace(0.2, 0.95, 16)]
#     target_neg_fp = float(getattr(args, "target_neg_fp", 0.2))
#     lambda_fp = float(getattr(args, "lambda_fp", 0.5))

#     # ===== 新增开关/权重 =====
#     use_cldice = bool(getattr(args, "use_cldice", True))
#     cldice_weight = float(getattr(args, "cldice_weight", 0.5))
#     cldice_iters  = int(getattr(args, "cldice_iters", 12))

#     use_centerline = bool(getattr(args, "use_centerline_head", True))
#     centerline_weight = float(getattr(args, "centerline_weight", 0.3))

#     use_hardneg = bool(getattr(args, "hardneg", True))
#     hardneg_start_epoch = int(getattr(args, "hardneg_start_epoch", 3))
#     hardneg_gamma = float(getattr(args, "hardneg_gamma", 5.0))
#     hardneg_max_weight = float(getattr(args, "hardneg_max_weight", 5.0))
#     hardneg_score_limit = int(getattr(args, "hardneg_score_limit", 0))

#     if _SKLEARN:
#         from sklearn.model_selection import StratifiedKFold
#         skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=args.rand_seed)
#         splits = skf.split(np.arange(len(pool_items)), np.array(strata_ids))
#     else:
#         by_sid = {}
#         for i, sid in enumerate(strata_ids):
#             by_sid.setdefault(sid, []).append(i)
#         for s in by_sid.values(): random.Random(args.rand_seed).shuffle(s)
#         def gen():
#             for f in range(K):
#                 val_idx = []
#                 for sid, idxs in by_sid.items(): val_idx += idxs[f::K]
#                 train_idx = sorted(list(set(range(len(pool_items))) - set(val_idx)))
#                 yield train_idx, val_idx
#         splits = gen()

#     per_fold_iou = []
#     start_fold = int(getattr(args, "start_fold", 1))

#     for fold, (tr_idx, va_idx) in enumerate(splits, 1):
#         if fold < start_fold:
#             print(f"\n----- Fold {fold}/{K} skipped (start_fold={start_fold}) -----")
#             per_fold_iou.append(0.0)
#             continue

#         print(f"\n----- Fold {fold}/{K} -----")
#         items_train = [pool_items[i] for i in tr_idx]
#         items_val   = [pool_items[i] for i in va_idx]

#         ds_train = CVPairDataset(items_train, transform_train, with_centerline=use_centerline)
#         ds_val   = CVPairDataset(items_val,   transform_val,   with_centerline=use_centerline)

#         base_neg_w = float(getattr(args, "neg_sample_weight", 0.6))
#         weights = build_sampler_for_subset(items_train, neg_sample_weight=base_neg_w)
#         sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

#         dl_train = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler,
#                               num_workers=args.workers, pin_memory=True, drop_last=True)
#         dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
#                               num_workers=args.workers, pin_memory=True, drop_last=False)

#         # Model
#         model = CenterlineModel() if use_centerline else BaseModel()

#         if len(args.gpus) == 0:
#             device = torch.device('cpu')
#         elif len(args.gpus) == 1:
#             torch.cuda.set_device(args.gpus[0]); model.cuda(); device = torch.device('cuda')
#         else:
#             gpus = ','.join(str(i) for i in args.gpus); os.environ["CUDA_VISIBLE_DEVICES"] = gpus
#             model.cuda(); model = torch.nn.DataParallel(model, device_ids=list(range(len(args.gpus)))); device = torch.device('cuda')

#         # BCE pos_weight（你当前固定值）
#         pos_weight_value = float(getattr(args, "pos_weight_value", 5.0))
#         bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))

#         # seg aux loss (tversky/dice)
#         use_tversky = bool(getattr(args, "use_tversky", True))
#         if use_tversky:
#             alpha = float(getattr(args, "tversky_alpha", 0.4))
#             beta  = float(getattr(args, "tversky_beta", 0.6))
#             aux_loss = smp.losses.TverskyLoss(mode='binary', from_logits=True, alpha=alpha, beta=beta)
#         else:
#             aux_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)

#         bg_loss_scale = float(getattr(args, "bg_loss_scale", 1.8))
#         cen_bce = nn.BCEWithLogitsLoss()

#         def compute_loss(model_out, tgt_mask, tgt_cen=None):
#             """
#             model_out:
#               - tensor logits (B,1,H,W) if no centerline head
#               - (seg_logits, cen_logits) if centerline head
#             tgt_mask: (B,1,H,W)
#             tgt_cen : (B,1,H,W) or None
#             """
#             if isinstance(model_out, (tuple, list)):
#                 seg_logits, cen_logits = model_out
#             else:
#                 seg_logits, cen_logits = model_out, None

#             b = bce(seg_logits, tgt_mask)
#             if tgt_mask.sum() == 0:
#                 seg_loss = bg_loss_scale * b
#             else:
#                 seg_loss = b + aux_loss(seg_logits, tgt_mask)

#             if use_cldice:
#                 seg_loss = seg_loss + cldice_weight * cldice_loss_from_logits(seg_logits, tgt_mask, iters=cldice_iters)

#             if (cen_logits is not None) and (tgt_cen is not None):
#                 cen_loss = cen_bce(cen_logits, tgt_cen)
#                 seg_loss = seg_loss + centerline_weight * cen_loss

#             return seg_loss

#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#         scheduler = get_cosine_schedule_with_warmup(
#             optimizer,
#             num_warmup_steps=args.warmup_epochs,
#             num_training_steps=args.epochs
#         )

#         use_amp = bool(getattr(args, 'amp', True)) and torch.cuda.is_available()
#         scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
#         ema = ModelEMA(model, decay=float(getattr(args, 'ema_decay', 0.999))) if bool(getattr(args, 'ema', True)) else None

#         patience = int(getattr(args, 'patience', 15))
#         patience_cnt, best_iou, best_thr, best_val_loss = 0, -1.0, 0.5, 1e9
#         best_score, best_neg_fp = -1e9, 1.0

#         best_iou_path  = os.path.join(args.MODEL_PATH, f"{args.model_name}_fp_fold{fold}_best_iou.pth")
#         best_loss_path = os.path.join(args.MODEL_PATH, f"{args.model_name}_fp_fold{fold}_best_loss.pth")
#         final_path     = os.path.join(args.MODEL_PATH, f"{args.model_name}_fp_fold{fold}_final.pth")

#         for epoch in range(1, args.epochs+1):
#             model.train()
#             losses = []

#             for batch in tqdm(dl_train):
#                 if len(batch) == 2:
#                     X, y = batch; c = None
#                 else:
#                     X, y, c = batch

#                 X = X.to(device, non_blocking=True)
#                 y = y.to(device, non_blocking=True)
#                 if c is not None: c = c.to(device, non_blocking=True)

#                 optimizer.zero_grad(set_to_none=True)
#                 with torch.cuda.amp.autocast(enabled=use_amp):
#                     out = model(X if not args.upsample else F.interpolate(X, scale_factor=2, mode='bicubic', align_corners=False))
#                     loss = compute_loss(
#                         out,
#                         y if not args.upsample else F.interpolate(y, scale_factor=2, mode='nearest'),
#                         None if c is None else (c if not args.upsample else F.interpolate(c, scale_factor=2, mode='nearest'))
#                     )

#                 scaler.scale(loss).backward()
#                 scaler.step(optimizer); scaler.update()
#                 if ema is not None: ema.update(model)
#                 losses.append(loss.item())

#             scheduler.step()
#             print(f"[Fold {fold}] [Train {epoch:03d}/{args.epochs:03d}] loss={float(np.mean(losses)):.5f}")

#             # ===== hard-neg reweight after some epochs =====
#             if use_hardneg and epoch >= hardneg_start_epoch:
#                 eval_model = ema.ema if ema is not None else model
#                 new_weights = update_hardneg_weights(
#                     eval_model, items_train, device,
#                     base_neg_weight=base_neg_w,
#                     hardneg_gamma=hardneg_gamma,
#                     max_weight=hardneg_max_weight,
#                     score_limit=hardneg_score_limit
#                 )
#                 sampler = WeightedRandomSampler(new_weights, num_samples=len(new_weights), replacement=True)
#                 dl_train = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler,
#                                       num_workers=args.workers, pin_memory=True, drop_last=True)
#                 print(f"[Fold {fold}] [HardNeg] sampler updated at epoch {epoch}")

#             # ===== validation =====
#             eval_model = ema.ema if ema is not None else model
#             val_loss, thr, pos_iou, neg_fp, mIoU_all = eval_on_loader(
#                 eval_model, dl_val, device, compute_loss,
#                 thr_list, use_amp=use_amp, upsample=args.upsample,
#                 target_neg_fp=target_neg_fp, lambda_fp=lambda_fp
#             )
#             print(f"[Fold {fold}] [Valid {epoch:03d}/{args.epochs:03d}] "
#                   f"loss={val_loss:.5f}, IoU(pos)={pos_iou:.4f} @thr={thr:.2f}, "
#                   f"Neg FP={neg_fp:.4f}, mIoU(all)={mIoU_all:.4f}")

#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss; torch.save(eval_model.state_dict(), best_loss_path)

#             score = pos_iou - lambda_fp * neg_fp
#             if score > best_score:
#                 best_score = score
#                 best_iou = pos_iou
#                 best_thr = thr
#                 best_neg_fp = neg_fp
#                 torch.save(eval_model.state_dict(), best_iou_path)
#                 patience_cnt = 0
#             else:
#                 patience_cnt += 1

#             if patience_cnt >= patience:
#                 print(f"[Fold {fold}] Early stopping at epoch {epoch}.")
#                 break

#         torch.save((ema.ema if ema is not None else model).state_dict(), final_path)
#         print(f"[Fold {fold}] best_IoU={best_iou:.4f} @thr={best_thr:.2f}, best_NegFP={best_neg_fp:.4f}")

#         per_fold_iou.append(best_iou)
#         del ds_train, ds_val, dl_train, dl_val
#         gc.collect()

#     os.makedirs(args.MODEL_PATH, exist_ok=True)
#     with open(os.path.join(args.MODEL_PATH, f"{args.model_name}_cv_fp_summary.json"), "w") as f:
#         json.dump({"per_fold_iou": per_fold_iou}, f, indent=2)

#     print("\n========== CV(FP+Topo+Center+HardNeg) Finished ==========")
#     print("Per-fold IoU(pos):", [round(x,4) for x in per_fold_iou])
#     return per_fold_iou
from __future__ import annotations
import os, cv2, gc, json, random, glob, warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import segmentation_models_pytorch as smp

try:
    from sklearn.model_selection import StratifiedKFold
    _SKLEARN = True
except Exception:
    StratifiedKFold = None
    _SKLEARN = False

from __init__ import *
from args import *
from data_aug import transform_train, transform_val
from models.timm_model import Model as BaseModel
from lrscheduler import *

warnings.filterwarnings("ignore")
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


# =========================
# Utils
# =========================
def read_mask_binary_uint8(p: str) -> np.ndarray:
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Error loading mask: {p}")
    if m.max() > 1:
        _, m = cv2.threshold(m, 127, 1, cv2.THRESH_BINARY)
    return m.astype(np.uint8)

def scan_pairs(base: str) -> List[Tuple[str, str]]:
    img_dir = os.path.join(base, 'image')
    lbl_dir = os.path.join(base, 'label')
    if not (os.path.isdir(img_dir) and os.path.isdir(lbl_dir)):
        return []
    img_paths = [p for p in glob.glob(os.path.join(img_dir, '*'))
                 if Path(p).suffix.lower() in IMG_EXTS]
    lbl_paths = [p for p in glob.glob(os.path.join(lbl_dir, '*'))
                 if Path(p).suffix.lower() in IMG_EXTS]
    imap = {Path(p).stem: p for p in img_paths}
    lmap = {Path(p).stem: p for p in lbl_paths}
    stems = sorted(set(imap) & set(lmap))
    return [(imap[s], lmap[s]) for s in stems]

def build_strata(items: List[Tuple[str,str]]) -> Tuple[List[int], Dict[int,str]]:
    strata_names = []
    bins = getattr(args, "pos_bins", [0.0, 0.01, 0.05, 1.01])
    for _, mpath in items:
        m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"Error loading mask: {mpath}")
        if m.sum() == 0:
            strata_names.append('neg')
        else:
            r = (m > 0).sum() / m.size
            if r <= bins[1]: strata_names.append('pos_small')
            elif r <= bins[2]: strata_names.append('pos_mid')
            else: strata_names.append('pos_large')
    uniq = {name:i for i,name in enumerate(sorted(set(strata_names)))}
    strata_ids = [uniq[n] for n in strata_names]
    id2name = {i:n for n,i in uniq.items()}
    return strata_ids, id2name


# =========================
# Centerline target (CPU skeleton)
# =========================
def skeletonize_binary(mask_np: np.ndarray) -> np.ndarray:
    mask = (mask_np > 0).astype(np.uint8)
    if mask.sum() == 0:
        return mask
    skel = np.zeros_like(mask)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = mask.copy()
    while True:
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    skel = (skel > 0).astype(np.uint8)
    return skel


# =========================
# clDice (soft skeleton, differentiable)
# =========================
def soft_erode(img):
    p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
    p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
    return torch.min(p1, p2)

def soft_dilate(img):
    return F.max_pool2d(img, 3, 1, 1)

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skel(img, iters=10):
    skel = F.relu(img - soft_open(img))
    for _ in range(iters):
        img = soft_erode(img)
        opened = soft_open(img)
        delta = F.relu(img - opened)
        skel = skel + F.relu(delta - skel * delta)
    return skel

def cldice_loss_from_logits(pred_logits, gt, iters=10, smooth=1.0):
    pred = torch.sigmoid(pred_logits)
    skel_pred = soft_skel(pred, iters)
    skel_gt   = soft_skel(gt, iters)

    dims = (1,2,3)
    tprec = (skel_pred * gt).sum(dims) / (skel_pred.sum(dims) + smooth)
    tsens = (skel_gt * pred).sum(dims) / (skel_gt.sum(dims) + smooth)

    cldice = (2 * tprec * tsens + smooth) / (tprec + tsens + smooth)
    return 1.0 - cldice.mean()


# =========================
# Dataset
# =========================
class CVPairDataset(Dataset):
    def __init__(self, items: List[Tuple[str,str]], transform, with_centerline: bool=False):
        self.items = items
        self.t = transform
        self.with_centerline = with_centerline

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        ip, mp = self.items[idx]
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        if img is None: raise RuntimeError(f'Error loading {ip}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        m = read_mask_binary_uint8(mp)

        out = self.t(image=img, mask=m)
        image = out['image']
        mask_t = out['mask']
        if not isinstance(mask_t, torch.Tensor):
            mask_t = torch.from_numpy(mask_t)
        mask = (mask_t.float() > 0.5).float().unsqueeze(0)

        if not self.with_centerline:
            return image, mask

        m_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)
        cen_np = skeletonize_binary(m_np)
        cen = torch.from_numpy(cen_np).float().unsqueeze(0)
        return image, mask, cen


# =========================
# EMA
# =========================
class ModelEMA:
    def __init__(self, model, decay=0.999):
        from copy import deepcopy
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters(): p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd:
                v.copy_(v*d + msd[k]*(1.0-d))


# =========================
# Multi-head model (seg + centerline)
# =========================
class CenterlineModel(BaseModel):
    def __init__(self, pretrained=True, decoder_channels=(384,192,96,64), out_indices=(0,1,2,3)):
        super().__init__(pretrained=pretrained, decoder_channels=decoder_channels, out_indices=out_indices)
        self.centerline_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=1, activation=None, kernel_size=3,
        )

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        seg_logits = self.segmentation_head(decoder_output)
        cen_logits = self.centerline_head(decoder_output)
        return seg_logits, cen_logits


# =========================
# Sampler (pos/neg base)
# =========================
def build_sampler_for_subset(items_subset: List[Tuple[str,str]], neg_sample_weight: float=0.6):
    pos_idx, neg_idx = [], []
    for i, (_, mpath) in enumerate(items_subset):
        m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if m is None: raise RuntimeError(f"Error loading mask: {mpath}")
        if m.sum() > 0: pos_idx.append(i)
        else: neg_idx.append(i)

    weights = [0.0]*len(items_subset)
    for i in pos_idx: weights[i] = 1.0
    for i in neg_idx: weights[i] = max(1e-6, float(neg_sample_weight))
    return weights


# =========================
# Hard-neg mining (epoch-wise reweight)
# =========================
@torch.no_grad()
def update_hardneg_weights(model,
                           items_train: List[Tuple[str,str]],
                           device,
                           base_neg_weight: float,
                           hardneg_gamma: float,
                           max_weight: float,
                           score_limit: int=0):
    ds_score = CVPairDataset(items_train, transform_val, with_centerline=False)
    dl_score = DataLoader(ds_score, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    neg_scores = []
    neg_indices = []

    model.eval()
    for bi, batch in enumerate(dl_score):
        X, y = batch
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(X)
        seg_logits = out[0] if isinstance(out, (tuple, list)) else out
        prob = torch.sigmoid(seg_logits)

        for i in range(prob.shape[0]):
            yi = y[i]
            if yi.sum() == 0:
                fp_score = float(prob[i].max().item())
                neg_scores.append(fp_score)
                neg_indices.append(bi*args.batch_size + i)

        if score_limit > 0 and len(neg_scores) >= score_limit:
            break

    base_weights = build_sampler_for_subset(items_train, base_neg_weight)
    for idx, fp_score in zip(neg_indices, neg_scores):
        w = base_neg_weight * (1.0 + hardneg_gamma * fp_score)
        w = float(min(max_weight, max(1e-6, w)))
        base_weights[idx] = w

    return base_weights


# =========================
# Eval
# =========================
def eval_on_loader(model, loader, device, compute_loss, thr_list,
                   use_amp=False, upsample=False,
                   target_neg_fp=0.2, lambda_fp=0.5):
    model.eval()
    losses = []
    pos_inter = {thr:0.0 for thr in thr_list}
    pos_union = {thr:0.0 for thr in thr_list}
    neg_ok   = {thr:0.0 for thr in thr_list}
    neg_cnt  = 0.0
    amp = bool(use_amp)

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                X, y = batch
                c = None
            else:
                X, y, c = batch

            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if c is not None:
                c = c.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp):
                out = model(X if not upsample else F.interpolate(X, scale_factor=2, mode='bicubic', align_corners=False))
                loss = compute_loss(
                    out,
                    y if not upsample else F.interpolate(y, scale_factor=2, mode='nearest'),
                    c if c is None else (c if not upsample else F.interpolate(c, scale_factor=2, mode='nearest'))
                )
            losses.append(loss.item())

            seg_logits = out[0] if isinstance(out, (tuple, list)) else out
            prob = torch.sigmoid(seg_logits)
            if upsample:
                prob = F.interpolate(prob, scale_factor=0.5, mode='nearest')

            for thr in thr_list:
                pb = (prob > thr).float()
                for pbi, gti in zip(pb, y):
                    if gti.sum() == 0:
                        neg_cnt += 1.0
                        neg_ok[thr] += 1.0 if pbi.sum() == 0 else 0.0
                    else:
                        inter = torch.logical_and(pbi>0.5, gti>0.5).sum().item()
                        union = torch.logical_or (pbi>0.5, gti>0.5).sum().item()
                        pos_inter[thr] += inter
                        pos_union[thr] += max(1.0, union)

    avg_loss = float(np.mean(losses)) if losses else 0.0

    best_thr, best_pos_iou, best_neg_fp = thr_list[0], 0.0, 1.0
    chosen = False
    for thr in thr_list:
        pos_iou = (pos_inter[thr] / pos_union[thr]) if pos_union[thr] > 0 else 0.0
        neg_fp  = 1.0 - (neg_ok[thr] / neg_cnt) if neg_cnt > 0 else 0.0
        if neg_fp <= target_neg_fp:
            if (not chosen) or (pos_iou > best_pos_iou):
                best_thr, best_pos_iou, best_neg_fp = thr, pos_iou, neg_fp
                chosen = True

    if not chosen:
        best_score = -1e9
        for thr in thr_list:
            pos_iou = (pos_inter[thr] / pos_union[thr]) if pos_union[thr] > 0 else 0.0
            neg_fp  = 1.0 - (neg_ok[thr] / neg_cnt) if neg_cnt > 0 else 0.0
            score = pos_iou - lambda_fp * neg_fp
            if score > best_score:
                best_score = score
                best_thr, best_pos_iou, best_neg_fp = thr, pos_iou, neg_fp

    mIoU_all = 0.0
    if pos_union[best_thr] + neg_cnt > 0:
        mIoU_all = (pos_inter[best_thr] + neg_ok[best_thr]) / (pos_union[best_thr] + neg_cnt)

    return avg_loss, best_thr, best_pos_iou, best_neg_fp, mIoU_all


# =========================
# Weight schedule helpers
# =========================
def ramp_linear(progress: float, start_ratio: float, max_weight: float) -> float:
    """progress: epoch/epochs in [0,1]. start_ratio in [0,1)."""
    if progress < start_ratio:
        return 0.0
    denom = max(1e-6, 1.0 - start_ratio)
    t = min(1.0, (progress - start_ratio) / denom)
    return max_weight * t


# =========================
# Trainer
# =========================
def trainer_cv_fp():
    print("========== 5-Fold CV (FP-suppression + clDice + centerline + hardneg + schedules) ==========")
    np.random.seed(args.rand_seed); random.seed(args.rand_seed); torch.manual_seed(args.rand_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.rand_seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

    pool_items = []
    for sub in ['train', 'val']:
        base = os.path.join(args.root, sub)
        if os.path.isdir(base): pool_items += scan_pairs(base)
    if len(pool_items) == 0:
        raise RuntimeError("No data found under root/train or root/val (image/label).")

    strata_ids, id2name = build_strata(pool_items)
    print("Strata:", {i:n for i,n in id2name.items()})

    K = int(getattr(args, "cv_folds", 5))
    thr_list = [round(x,2) for x in np.linspace(0.2, 0.95, 16)]
    target_neg_fp = float(getattr(args, "target_neg_fp", 0.2))
    lambda_fp = float(getattr(args, "lambda_fp", 0.5))

    # ===== static toggles & max weights =====
    use_cldice = bool(getattr(args, "use_cldice", True))
    cldice_weight_max = float(getattr(args, "cldice_weight", 0.5))
    cldice_iters  = int(getattr(args, "cldice_iters", 12))

    use_centerline = bool(getattr(args, "use_centerline_head", True))
    centerline_weight_max = float(getattr(args, "centerline_weight", 0.3))

    use_hardneg = bool(getattr(args, "hardneg", True))
    hardneg_start_epoch = int(getattr(args, "hardneg_start_epoch", 3))
    hardneg_gamma = float(getattr(args, "hardneg_gamma", 5.0))
    hardneg_max_weight = float(getattr(args, "hardneg_max_weight", 5.0))
    hardneg_score_limit = int(getattr(args, "hardneg_score_limit", 0))

    # ===== schedule ratios from args =====
    cldice_start_ratio = float(getattr(args, "cldice_start_ratio", 0.2))
    cen_start_ratio    = float(getattr(args, "cen_start_ratio", 0.2))
    hardneg_start_ratio= float(getattr(args, "hardneg_start_ratio", 0.35))

    if _SKLEARN:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=args.rand_seed)
        splits = skf.split(np.arange(len(pool_items)), np.array(strata_ids))
    else:
        by_sid = {}
        for i, sid in enumerate(strata_ids):
            by_sid.setdefault(sid, []).append(i)
        for s in by_sid.values(): random.Random(args.rand_seed).shuffle(s)
        def gen():
            for f in range(K):
                val_idx = []
                for sid, idxs in by_sid.items(): val_idx += idxs[f::K]
                train_idx = sorted(list(set(range(len(pool_items))) - set(val_idx)))
                yield train_idx, val_idx
        splits = gen()

    per_fold_iou = []
    start_fold = int(getattr(args, "start_fold", 1))

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        if fold < start_fold:
            print(f"\n----- Fold {fold}/{K} skipped (start_fold={start_fold}) -----")
            per_fold_iou.append(0.0)
            continue

        print(f"\n----- Fold {fold}/{K} -----")
        items_train = [pool_items[i] for i in tr_idx]
        items_val   = [pool_items[i] for i in va_idx]

        ds_train = CVPairDataset(items_train, transform_train, with_centerline=use_centerline)
        ds_val   = CVPairDataset(items_val,   transform_val,   with_centerline=use_centerline)

        base_neg_w = float(getattr(args, "neg_sample_weight", 0.6))
        weights = build_sampler_for_subset(items_train, neg_sample_weight=base_neg_w)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

        dl_train = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
        dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True, drop_last=False)

        model = CenterlineModel() if use_centerline else BaseModel()

        if len(args.gpus) == 0:
            device = torch.device('cpu')
        elif len(args.gpus) == 1:
            torch.cuda.set_device(args.gpus[0]); model.cuda(); device = torch.device('cuda')
        else:
            gpus = ','.join(str(i) for i in args.gpus); os.environ["CUDA_VISIBLE_DEVICES"] = gpus
            model.cuda(); model = torch.nn.DataParallel(model, device_ids=list(range(len(args.gpus)))); device = torch.device('cuda')

        pos_weight_value = float(getattr(args, "pos_weight_value", 5.0))
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))

        use_tversky = bool(getattr(args, "use_tversky", True))
        if use_tversky:
            alpha = float(getattr(args, "tversky_alpha", 0.4))
            beta  = float(getattr(args, "tversky_beta", 0.6))
            aux_loss = smp.losses.TverskyLoss(mode='binary', from_logits=True, alpha=alpha, beta=beta)
        else:
            aux_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)

        bg_loss_scale = float(getattr(args, "bg_loss_scale", 1.8))
        cen_bce = nn.BCEWithLogitsLoss()

        # ===== current (scheduled) weights, will be updated per epoch =====
        cur_use_cldice = False
        cur_cldice_w = 0.0
        cur_cen_w = 0.0
        cur_use_hardneg = False

        def compute_loss(model_out, tgt_mask, tgt_cen=None):
            if isinstance(model_out, (tuple, list)):
                seg_logits, cen_logits = model_out
            else:
                seg_logits, cen_logits = model_out, None

            b = bce(seg_logits, tgt_mask)
            if tgt_mask.sum() == 0:
                seg_loss = bg_loss_scale * b
            else:
                seg_loss = b + aux_loss(seg_logits, tgt_mask)

            if cur_use_cldice and cur_cldice_w > 0:
                seg_loss = seg_loss + cur_cldice_w * cldice_loss_from_logits(
                    seg_logits, tgt_mask, iters=cldice_iters
                )

            if (cen_logits is not None) and (tgt_cen is not None) and cur_cen_w > 0:
                cen_loss = cen_bce(cen_logits, tgt_cen)
                seg_loss = seg_loss + cur_cen_w * cen_loss

            return seg_loss

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_epochs,
            num_training_steps=args.epochs
        )

        use_amp = bool(getattr(args, 'amp', True)) and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        ema = ModelEMA(model, decay=float(getattr(args, 'ema_decay', 0.999))) if bool(getattr(args, 'ema', True)) else None

        patience = int(getattr(args, 'patience', 15))
        patience_cnt, best_iou, best_thr, best_val_loss = 0, -1.0, 0.5, 1e9
        best_score, best_neg_fp = -1e9, 1.0

        best_iou_path  = os.path.join(args.MODEL_PATH, f"{args.model_name}_fp_fold{fold}_best_iou.pth")
        best_loss_path = os.path.join(args.MODEL_PATH, f"{args.model_name}_fp_fold{fold}_best_loss.pth")
        final_path     = os.path.join(args.MODEL_PATH, f"{args.model_name}_fp_fold{fold}_final.pth")

        for epoch in range(1, args.epochs+1):
            # ===== update scheduled weights / toggles =====
            progress = epoch / float(args.epochs)

            cur_cldice_w = ramp_linear(progress, cldice_start_ratio, cldice_weight_max)
            cur_use_cldice = use_cldice and (cur_cldice_w > 0)

            cur_cen_w = ramp_linear(progress, cen_start_ratio, centerline_weight_max)

            cur_use_hardneg = use_hardneg and (progress >= hardneg_start_ratio)

            if epoch == 1 or epoch % 5 == 0:
                print(f"[Fold {fold}] schedule @epoch{epoch}: "
                      f"clDice_w={cur_cldice_w:.3f}, cen_w={cur_cen_w:.3f}, hardneg={cur_use_hardneg}")

            model.train()
            losses = []

            for batch in tqdm(dl_train):
                if len(batch) == 2:
                    X, y = batch; c = None
                else:
                    X, y, c = batch

                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if c is not None: c = c.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(X if not args.upsample else F.interpolate(
                        X, scale_factor=2, mode='bicubic', align_corners=False))
                    loss = compute_loss(
                        out,
                        y if not args.upsample else F.interpolate(y, scale_factor=2, mode='nearest'),
                        None if c is None else (c if not args.upsample else F.interpolate(c, scale_factor=2, mode='nearest'))
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
                if ema is not None: ema.update(model)
                losses.append(loss.item())

            scheduler.step()
            print(f"[Fold {fold}] [Train {epoch:03d}/{args.epochs:03d}] loss={float(np.mean(losses)):.5f}")

            # ===== hard-neg reweight after some epochs =====
            if cur_use_hardneg and epoch >= hardneg_start_epoch:
                eval_model = ema.ema if ema is not None else model
                new_weights = update_hardneg_weights(
                    eval_model, items_train, device,
                    base_neg_weight=base_neg_w,
                    hardneg_gamma=hardneg_gamma,
                    max_weight=hardneg_max_weight,
                    score_limit=hardneg_score_limit
                )
                sampler = WeightedRandomSampler(new_weights, num_samples=len(new_weights), replacement=True)
                dl_train = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler,
                                      num_workers=args.workers, pin_memory=True, drop_last=True)
                print(f"[Fold {fold}] [HardNeg] sampler updated at epoch {epoch}")

            # ===== validation =====
            eval_model = ema.ema if ema is not None else model
            val_loss, thr, pos_iou, neg_fp, mIoU_all = eval_on_loader(
                eval_model, dl_val, device, compute_loss,
                thr_list, use_amp=use_amp, upsample=args.upsample,
                target_neg_fp=target_neg_fp, lambda_fp=lambda_fp
            )
            print(f"[Fold {fold}] [Valid {epoch:03d}/{args.epochs:03d}] "
                  f"loss={val_loss:.5f}, IoU(pos)={pos_iou:.4f} @thr={thr:.2f}, "
                  f"Neg FP={neg_fp:.4f}, mIoU(all)={mIoU_all:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss; torch.save(eval_model.state_dict(), best_loss_path)

            score = pos_iou - lambda_fp * neg_fp
            if score > best_score:
                best_score = score
                best_iou = pos_iou
                best_thr = thr
                best_neg_fp = neg_fp
                torch.save(eval_model.state_dict(), best_iou_path)
                patience_cnt = 0
            else:
                patience_cnt += 1

            if patience_cnt >= patience:
                print(f"[Fold {fold}] Early stopping at epoch {epoch}.")
                break

        torch.save((ema.ema if ema is not None else model).state_dict(), final_path)
        print(f"[Fold {fold}] best_IoU={best_iou:.4f} @thr={best_thr:.2f}, best_NegFP={best_neg_fp:.4f}")

        per_fold_iou.append(best_iou)
        del ds_train, ds_val, dl_train, dl_val
        gc.collect()

    os.makedirs(args.MODEL_PATH, exist_ok=True)
    with open(os.path.join(args.MODEL_PATH, f"{args.model_name}_cv_fp_summary.json"), "w") as f:
        json.dump({"per_fold_iou": per_fold_iou}, f, indent=2)

    print("\n========== CV(FP+Topo+Center+HardNeg+Schedule) Finished ==========")
    print("Per-fold IoU(pos):", [round(x,4) for x in per_fold_iou])
    return per_fold_iou
