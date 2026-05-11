import logging
import os
from abc import abstractmethod

import cv2
import numpy as np
import spacy
# import scispacy
import torch
import re
from modules.utils import generate_heatmap,generate_dynamic_nodes
from pathlib import Path


class BaseTester(object):
    def __init__(self, model, criterion, metric_ftns, args):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir

        self._load_checkpoint(args.load)

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _load_checkpoint(self, load_path):


        load_path = r'\model_best.pth'

        load_path = str(load_path)
        self.logger.info("Loading checkpoint: {} ...".format(load_path))
        checkpoint = torch.load(load_path,map_location='cpu')


        self.model.load_state_dict(checkpoint['state_dict'],strict=False)

def sanitize_filename(word):

    return re.sub(r'[^\w_]', '', word)
class Tester(BaseTester):
    def __init__(self, model, criterion, metric_ftns, args, test_dataloader):
        super(Tester, self).__init__(model, criterion, metric_ftns, args)
        self.test_dataloader = test_dataloader



    def test(self):
        self.logger.info('Start to evaluate in the test set.')
        self.model.eval()
        log = dict()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())

                path_ = r'./'

                with open(path_+str(images_id)[2:-3]+'.txt','w') as f:
                            f.write(str(reports[0]))
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            print(log)
        return log


    def plot(self):
        assert self.args.batch_size == 1 and self.args.beam_size == 1
        self.logger.info('begin')

        # 创建保存目录
        save_dirs = ["heatmaps", "dynamic_nodes"]
        for dir_name in save_dirs:
            os.makedirs(os.path.join(self.save_dir, dir_name), exist_ok=True)



        # 图像反归一化参数
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images = images.to(self.device)

                # 处理images_id，提取需要的部分
                image_id_str = str(images_id)[2:-3]
                safe_image_id = re.sub(r'[^\w-]', '', image_id_str)[:50]  # 限制长度并移除非法字符

                # === 1. 获取分块坐标和注意力权重 ===
                _, _, grid_coords = self.model.visual_extractor(images)
                grid_coords = grid_coords.squeeze(0).cpu().numpy().astype(int)  # (num_patches, 2)

                # === 2. 处理原始图像 ===
                image_tensor = images[0].cpu() * std + mean
                image_np = image_tensor.clamp(0, 1).numpy().transpose(1, 2, 0)  # (H, W, C)
                image_np = (image_np * 255).astype(np.uint8)
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                # === 3. 生成两种可视化结果 ===
                output, _ = self.model(images, mode='sample')
                report = self.model.tokenizer.decode_batch(output.cpu().numpy())[0].split()
                attention_weights = self.model.encoder_decoder.attention_weights[:-1]

                for word_idx, (attns, word) in enumerate(zip(attention_weights, report)):
                    safe_word = re.sub(r'[^\w-]', '', word)[:50]
                    for layer_idx, attn in enumerate(attns):
                        attn_np = attn.mean(1).squeeze()

                        # 跳过无效权重
                        if attn_np.size != 49:
                            continue

                        # --- 生成纯热力图 ---
                        heatmap = generate_heatmap(image_bgr, attn_np)
                        heatmap_dir = os.path.join(self.save_dir, "heatmaps", safe_image_id, f"layer_{layer_idx}")
                        Path(heatmap_dir).mkdir(parents=True, exist_ok=True)
                        heatmap_path = os.path.join(heatmap_dir, f"{word_idx:04d}_{safe_word}.png")
                        cv2.imwrite(heatmap_path, heatmap)

                        # --- 生成动态节点图 ---
                        node_image = generate_dynamic_nodes(image_bgr, attn_np, grid_coords)
                        node_dir = os.path.join(self.save_dir, "dynamic_nodes", safe_image_id,f"layer_{layer_idx}")
                        Path(node_dir).mkdir(parents=True, exist_ok=True)
                        node_path = os.path.join(node_dir, f"{word_idx:04d}_{safe_word}.png")
                        cv2.imwrite(node_path, node_image)

                        self.logger.info(f"已保存: {heatmap_path} 和 {node_path}")