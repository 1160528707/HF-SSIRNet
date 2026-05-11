# plot_one.py
import argparse
import copy
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

import models.models
from modules.dataloaders import R2DataLoader
from modules.loss import compute_loss
from modules.metrics import compute_scores
from modules.tokenizers import Tokenizer


def prepare_device(n_gpu_use: int):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        logging.warning("No GPU found, using CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        logging.warning(f"n_gpu_use={n_gpu_use} but only {n_gpu} GPUs available.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    device_ids = list(range(n_gpu_use))
    return device, device_ids


def normalize_image_id(x):
    s = str(x).strip()
    s = s.replace("\\", "/")
    s = os.path.basename(s)
    stem, _ = os.path.splitext(s)
    if stem.strip():
        s = stem
    return s


def extract_id_from_item(item):
    if not isinstance(item, dict):
        return None
    for key in ["id", "image_id", "images_id", "study_id", "dicom_id"]:
        if key in item:
            return normalize_image_id(item[key])
    if "image_path" in item:
        return normalize_image_id(item["image_path"])
    if "path" in item:
        return normalize_image_id(item["path"])
    return None


def filter_items(items, target_ids):
    kept = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = extract_id_from_item(item)
        if item_id in target_ids:
            kept.append(item)
    return kept


def build_filtered_annotation(original_ann_path, target_ids, save_dir):
    """
    兼容：
    1) {"train":[...], "val":[...], "test":[...]}
    2) {"6037": ...}
    3) 其他嵌套 dict/list
    """
    with open(original_ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def filter_any(obj):
        if isinstance(obj, list):
            kept = filter_items(obj, target_ids)
            return kept, len(kept)

        if isinstance(obj, dict):
            # 先尝试 key 本身就是 image id
            direct_key_match = {}
            for k, v in obj.items():
                nk = normalize_image_id(k)
                if nk in target_ids:
                    direct_key_match[k] = v
            if len(direct_key_match) > 0:
                return direct_key_match, len(direct_key_match)

            filtered_dict = {}
            total = 0
            for k, v in obj.items():
                if isinstance(v, (list, dict)):
                    sub_filtered, sub_count = filter_any(v)
                    filtered_dict[k] = sub_filtered
                    total += sub_count
                else:
                    filtered_dict[k] = v
            return filtered_dict, total

        return obj, 0

    filtered, total_kept = filter_any(data)

    if total_kept == 0:
        raise ValueError(
            f"在标注文件里没有找到目标图像: {sorted(target_ids)}\n"
            f"请检查 target_image_ids 或 ann_path 的字段格式。"
        )

    temp_ann_path = os.path.join(save_dir, "filtered_annotation_plot_one.json")
    with open(temp_ann_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    return temp_ann_path, total_kept


def find_image_file(image_dir, image_id):
    """
    在 image_dir 里递归寻找 image_id 对应的图片。
    """
    image_id = normalize_image_id(image_id)
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]

    # 先直接尝试顶层
    for ext in exts:
        p = os.path.join(image_dir, image_id + ext)
        if os.path.exists(p):
            return p

    # 递归搜索
    for root, _, files in os.walk(image_dir):
        for f in files:
            stem, ext = os.path.splitext(f)
            if stem == image_id and ext.lower() in exts:
                return os.path.join(root, f)

    return None


def tensor_to_bgr_image(img_tensor):
    """
    把 dataloader 输出的 tensor 转成可视化 BGR 图。
    支持:
    - [C,H,W]
    - [1,C,H,W]
    """
    if torch.is_tensor(img_tensor):
        img = img_tensor.detach().cpu().float().numpy()
    else:
        img = np.asarray(img_tensor)

    if img.ndim == 4:
        img = img[0]

    if img.ndim != 3:
        raise ValueError(f"无法处理的图像维度: {img.shape}")

    # CHW -> HWC
    if img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))

    # 归一化到 0~255
    img = img.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)

    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    img = (img * 255).clip(0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_bgr


def make_grid_coords(h, w, grid_h=7, grid_w=7):
    ys = np.linspace(h / (2 * grid_h), h - h / (2 * grid_h), grid_h)
    xs = np.linspace(w / (2 * grid_w), w - w / (2 * grid_w), grid_w)
    coords = []
    for y in ys:
        for x in xs:
            coords.append((int(x), int(y)))
    return coords


def reduce_attention_to_49(attn_np):
    """
    把注意力长度归一化到 49:
    - 若本来就是 49，直接返回
    - 若是 98/147/196... 这种 49 的整数倍，则 reshape 后求平均
    """
    attn_np = np.asarray(attn_np).reshape(-1)
    n = attn_np.size

    if n == 49:
        return attn_np

    if n % 49 == 0:
        attn_np = attn_np.reshape(n // 49, 49).mean(axis=0)
        return attn_np

    return None


def generate_heatmap(image_bgr, attn_np):
    attn_49 = reduce_attention_to_49(attn_np)
    if attn_49 is None:
        return None

    h, w = image_bgr.shape[:2]
    heat = attn_49.reshape(7, 7).astype(np.float32)
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC)

    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / heat.max()

    heat_u8 = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image_bgr, 0.55, heat_color, 0.45, 0)
    return overlay


def generate_dynamic_nodes(image_bgr, attn_np, grid_coords):
    attn_49 = reduce_attention_to_49(attn_np)
    if attn_49 is None:
        return None

    attn_49 = attn_49.astype(np.float32)
    attn_49 = attn_49 - attn_49.min()
    if attn_49.max() > 0:
        attn_49 = attn_49 / attn_49.max()

    canvas = image_bgr.copy()

    for (x, y), a in zip(grid_coords, attn_49):
        radius = int(4 + 16 * float(a))
        color = (0, int(255 * (1 - a)), int(255 * a))  # BGR
        cv2.circle(canvas, (x, y), radius, color, thickness=-1)
        cv2.circle(canvas, (x, y), radius, (255, 255, 255), thickness=1)

    return canvas


class BaseTester(object):
    def __init__(self, model, criterion, metric_ftns, args):
        self.args = args
        self.logger = logging.getLogger(__name__)

        self.device, device_ids = prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns

        self._load_checkpoint(args.load)

    def _load_checkpoint(self, load_path):
        self.logger.info(f"Loading checkpoint: {load_path} ...")
        checkpoint = torch.load(load_path, map_location=self.device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        self.logger.info("Checkpoint loaded.")
        self.logger.info(f"Missing keys: {len(missing)}")
        self.logger.info(f"Unexpected keys: {len(unexpected)}")


class Tester(BaseTester):
    def __init__(self, model, criterion, metric_ftns, args, test_dataloader):
        super(Tester, self).__init__(model, criterion, metric_ftns, args)
        self.test_dataloader = test_dataloader
        self.model.eval()
        self.save_dir = os.path.join(args.save_dir, args.exp_name)
        os.makedirs(self.save_dir, exist_ok=True)

    @torch.no_grad()
    def plot(self):
        assert self.args.batch_size == 1
        self.logger.info("开始生成热力图和动态节点图...")

        processed_target_ids = {normalize_image_id(x) for x in self.args.target_image_ids.split(",") if x.strip()}
        finished_ids = set()

        # 用于对照查看报告
        compare_dir = os.path.join(self.save_dir, "report_compare")
        os.makedirs(compare_dir, exist_ok=True)

        for batch_idx, batch in enumerate(self.test_dataloader):
            if not isinstance(batch, (list, tuple)):
                raise ValueError(f"dataloader 返回的 batch 不是 tuple/list，而是: {type(batch)}")

            if len(batch) < 2:
                raise ValueError(f"dataloader 返回字段数太少: len(batch)={len(batch)}")

            # 兼容不同 dataloader 返回格式
            images_id = batch[0]
            images = batch[1]

            reports_ids = batch[2] if len(batch) > 2 else None
            reports_masks = batch[3] if len(batch) > 3 else None
            # 兼容 batch=1
            if isinstance(images_id, (list, tuple)):
                current_id = normalize_image_id(images_id[0])
            elif torch.is_tensor(images_id):
                current_id = normalize_image_id(images_id[0].item())
            else:
                current_id = normalize_image_id(images_id)

            if current_id not in processed_target_ids:
                continue

            self.logger.info(f"正在处理图像: {current_id}")

            images = images.to(self.device)

            # === 1. 当前 forward 生成报告（和 test_rl_to_txt.py 对齐） ===
            seq, extra = self.model(images, mode="sample")
            tokenizer = self.model.module.tokenizer if isinstance(self.model, torch.nn.DataParallel) else self.model.tokenizer
            report_text = tokenizer.decode_batch(seq.detach().cpu().numpy())[0].strip()
            report_words = report_text.split()

            with open(os.path.join(compare_dir, f"{current_id}_heatmap.txt"), "w", encoding="utf-8") as f:
                f.write(report_text)

            # === 2. 如果你已有保存好的报告，也一并拷出来方便核对 ===
            pred_report_dir = self.args.pred_report_dir
            if pred_report_dir and os.path.isdir(pred_report_dir):
                src_saved = os.path.join(pred_report_dir, f"{current_id}.txt")
                dst_saved = os.path.join(compare_dir, f"{current_id}_saved.txt")
                if os.path.exists(src_saved):
                    with open(src_saved, "r", encoding="utf-8") as f:
                        saved_text = f.read().strip()
                    with open(dst_saved, "w", encoding="utf-8") as f:
                        f.write(saved_text)
                else:
                    with open(dst_saved, "w", encoding="utf-8") as f:
                        f.write("[NOT FOUND]")

            # === 3. 取注意力 ===
            enc_dec = self.model.module.encoder_decoder if isinstance(self.model, torch.nn.DataParallel) else self.model.encoder_decoder
            if not hasattr(enc_dec, "attention_weights"):
                self.logger.error("模型中没有 attention_weights，无法生成热力图")
                continue

            attention_weights = enc_dec.attention_weights[:-1]
            if len(attention_weights) == 0:
                self.logger.warning(f"{current_id} 的 attention_weights 为空，跳过")
                continue

            # === 4. 底图 ===
            image_bgr = tensor_to_bgr_image(images[0])
            h, w = image_bgr.shape[:2]
            grid_coords = make_grid_coords(h, w, 7, 7)

            # === 5. 按词逐个保存 ===
            n = min(len(attention_weights), len(report_words))
            if len(attention_weights) != len(report_words):
                self.logger.warning(
                    f"{current_id}: attention_steps={len(attention_weights)}, "
                    f"report_words={len(report_words)}，按前 {n} 个对齐"
                )

            for word_idx in range(n):
                attns = attention_weights[word_idx]
                word = report_words[word_idx]

                safe_word = re.sub(r"[^\w-]", "", word)[:50]
                if len(safe_word) == 0:
                    safe_word = "token"

                for layer_idx, attn in enumerate(attns):
                    attn_np = attn.mean(1).squeeze()
                    if isinstance(attn_np, torch.Tensor):
                        attn_np = attn_np.detach().cpu().numpy()

                    reduced = reduce_attention_to_49(attn_np)
                    if reduced is None:
                        self.logger.warning(
                            f"跳过 {current_id} - word={safe_word} - layer={layer_idx}, "
                            f"因为注意力长度是 {np.asarray(attn_np).size}，无法归并到49"
                        )
                        continue

                    heatmap = generate_heatmap(image_bgr, reduced)
                    node_image = generate_dynamic_nodes(image_bgr, reduced, grid_coords)

                    heatmap_dir = os.path.join(self.save_dir, "heatmaps", current_id, f"layer_{layer_idx}")
                    node_dir = os.path.join(self.save_dir, "dynamic_nodes", current_id, f"layer_{layer_idx}")
                    Path(heatmap_dir).mkdir(parents=True, exist_ok=True)
                    Path(node_dir).mkdir(parents=True, exist_ok=True)

                    heatmap_path = os.path.join(heatmap_dir, f"{word_idx:04d}_{safe_word}.png")
                    node_path = os.path.join(node_dir, f"{word_idx:04d}_{safe_word}.png")

                    cv2.imwrite(heatmap_path, heatmap)
                    cv2.imwrite(node_path, node_image)

                    self.logger.info(f"已保存: {heatmap_path}")
                    self.logger.info(f"已保存: {node_path}")

            finished_ids.add(current_id)

            if finished_ids >= processed_target_ids:
                self.logger.info("目标图像已全部处理完成，提前结束。")
                break


def parse_args():
    parser = argparse.ArgumentParser()

    # ===== 和 test_rl_to_txt.py 对齐 =====
    parser.add_argument('--image_dir', type=str, default=r"E:\images")
    parser.add_argument('--ann_path', type=str, default=r"E:\dataset_fixed.json")

    parser.add_argument('--dataset_name', type=str, default='mimic_cxr', choices=['iu_xray', 'mimic_cxr'])
    parser.add_argument('--max_seq_length', type=int, default=60)
    parser.add_argument('--threshold', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--visual_extractor', type=str, default='vig_b_224')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True)

    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--d_vf', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--logit_layers', type=int, default=1)
    parser.add_argument('--bos_idx', type=int, default=0)
    parser.add_argument('--eos_idx', type=int, default=0)
    parser.add_argument('--pad_idx', type=int, default=0)
    parser.add_argument('--use_bn', type=int, default=0)
    parser.add_argument('--drop_prob_lm', type=float, default=0.5)
    parser.add_argument('--mode', type=str, default='test')

    parser.add_argument('--topk', type=int, default=32)
    parser.add_argument('--cmm_size', type=int, default=2048)
    parser.add_argument('--cmm_dim', type=int, default=512)

    # ===== 和 test_rl_to_txt.py 完全对齐 =====
    parser.add_argument('--sample_method', type=str, default='beam_search')
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sample_n', type=int, default=1)
    parser.add_argument('--group_size', type=int, default=1)
    parser.add_argument('--output_logsoftmax', type=int, default=1)
    parser.add_argument('--decoding_constraint', type=int, default=0)
    parser.add_argument('--block_trigrams', type=int, default=1)

    parser.add_argument('--n_gpu', type=int, default=1)


    parser.add_argument('--load', type=str,
                        default=r'\model_best.pth',
                        help='path of checkpoint')

    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])

    parser.add_argument('--save_dir', type=str,
                        default=r'\plot_one')
    parser.add_argument('--record_dir', type=str,
                        default=r'\plot_one')

    parser.add_argument('--log_period', type=int, default=1000)
    parser.add_argument('--save_period', type=int, default=1)
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'])
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4')
    parser.add_argument('--early_stop', type=int, default=50)

    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr_ve', type=float, default=5e-5)
    parser.add_argument('--lr_ed', type=float, default=7e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.98))
    parser.add_argument('--adam_eps', type=float, default=1e-9)
    parser.add_argument('--amsgrad', type=bool, default=True)
    parser.add_argument('--noamopt_warmup', type=int, default=5000)
    parser.add_argument('--noamopt_factor', type=int, default=1)

    parser.add_argument('--lr_scheduler', type=str, default='StepLR')
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.1)

    parser.add_argument('--seed', type=int, default=9233)

    parser.add_argument('--target_image_ids', type=str, default='6037',
                        help='comma separated image ids, e.g. 6037,33149')
    parser.add_argument('--exp_name', type=str, default='hfcvq',
                        help='name of current experiment')

    # 可选：把你之前生成好的 txt 抄进 report_compare 方便核对
    parser.add_argument('--pred_report_dir', type=str,
                        default=r'\\')

    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )

    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.record_dir, exist_ok=True)

    target_ids = {normalize_image_id(x) for x in args.target_image_ids.split(",") if x.strip()}

    filtered_ann_path, kept_num = build_filtered_annotation(
        original_ann_path=args.ann_path,
        target_ids=target_ids,
        save_dir=args.save_dir
    )
    print(f"Filtered annotation saved to: {filtered_ann_path}")
    print(f"Number of matched samples: {kept_num}")

    # 关键：Tokenizer 继续使用完整 ann_path，不能用过滤后的小 json
    tokenizer = Tokenizer(args)

    # 关键：只有 dataloader 用过滤后的小 json
    loader_args = copy.deepcopy(args)
    loader_args.ann_path = filtered_ann_path

    test_dataloader = R2DataLoader(loader_args, tokenizer, split=args.split, shuffle=False)

    model = models.models.BaseCMNModel(args, tokenizer)

    criterion = compute_loss
    metrics = compute_scores

    tester = Tester(model, criterion, metrics, args, test_dataloader)
    tester.plot()


if __name__ == '__main__':
    main()