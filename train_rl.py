import argparse
import logging
import os
import re
from abc import abstractmethod
from contextlib import contextmanager
from typing import List, Tuple, Optional

import numpy as np
import torch
from numpy import inf

import models.models
from modules.dataloaders import R2DataLoader
from modules.loss import compute_loss
import modules.metrics
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.tokenizers import Tokenizer


# -------------------------
# 一些小工具
# -------------------------
def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """兼容 DataParallel。"""
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _normalize_text(s: str) -> str:
    s = s.lower()
    # 统一空白与标点（尽量鲁棒，不依赖具体tokenizer）
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_keyword_groups(spec: str) -> List[List[str]]:

    groups: List[List[str]] = []
    for g in (spec or "").split(","):
        g = g.strip()
        if not g:
            continue
        aliases = [a.strip().lower() for a in g.split("|") if a.strip()]
        if aliases:
            groups.append(aliases)
    return groups


def compute_keyword_reward(texts: List[str], keyword_groups: List[List[str]]) -> Tuple[np.ndarray, np.ndarray]:

    n = len(texts)
    if len(keyword_groups) == 0:
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)

    rewards = np.zeros(n, dtype=np.float32)
    all_hit = np.zeros(n, dtype=np.float32)

    for i, t in enumerate(texts):
        t_norm = _normalize_text(t)
        hit = 0
        for group in keyword_groups:
            if any(alias in t_norm for alias in group):
                hit += 1
        rewards[i] = hit / float(len(keyword_groups))
        all_hit[i] = 1.0 if hit == len(keyword_groups) else 0.0
    return rewards, all_hit


def build_eos_mask(token_ids: torch.Tensor, eos_idx: int, pad_idx: int) -> torch.Tensor:

    bsz, T = token_ids.size()
    mask = (token_ids != pad_idx).float()
    if eos_idx is None:
        return mask

    eos_pos = torch.full((bsz,), T, device=token_ids.device, dtype=torch.long)
    eos_hits = (token_ids == eos_idx)
    if eos_hits.any():
        for i in range(bsz):
            idxs = torch.nonzero(eos_hits[i], as_tuple=False).view(-1)
            if idxs.numel() > 0:
                eos_pos[i] = idxs[0].item()

    ar = torch.arange(T, device=token_ids.device).unsqueeze(0).expand(bsz, T)
    after = ar > eos_pos.unsqueeze(1)
    mask = mask * (~after).float()
    return mask

class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max', 'off']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if getattr(args, "resume", None) is not None:
            self._resume_checkpoint(args.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            for key, value in log.items():
                self.logger.info('\t{:20s}: {}'.format(str(key), value))

            best = False
            if self.mnt_mode != 'off':
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. Monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation didn't improve for {} epochs. Stop.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

        self._print_best()

    def _record_best(self, log):
        if self.mnt_mode == 'off':
            return

        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        if self.mnt_mode == 'off':
            return
        self.logger.info('Best results (w.r.t {}) in val:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            self.logger.info('\t{:20s}: {}'.format(str(key), value))

        self.logger.info('Best results (w.r.t {}) in test:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            self.logger.info('\t{:20s}: {}'.format(str(key), value))

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("No GPU found, using CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("n_gpu_use={} but only {} GPUs available.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'rl_checkpoint_epoch_{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        # 兼容你原来的 current_checkpoint.pth
        current = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, current)

        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location='cpu')
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.mnt_best = checkpoint.get('monitor_best', self.mnt_best)
        self.model.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception as e:
                self.logger.warning("Optimizer state not loaded: {}".format(e))
        self.logger.info("Checkpoint loaded. Resume from epoch {}".format(self.start_epoch))


# -------------------------
# 强化学习 Trainer（SCST / REINFORCE）
# -------------------------
class RLTrainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler,
                 train_dataloader, val_dataloader, test_dataloader, keyword_groups):
        super().__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.keyword_groups = keyword_groups

        self.rl_weight = args.rl_weight
        self.ce_weight = args.ce_weight
        self.entropy_weight = args.entropy_weight
        self.grad_clip = args.grad_clip

        # token ids（通常 Tokenizer 会把这些写回 args）
        self.pad_idx = getattr(args, "pad_idx", 0)
        self.bos_idx = getattr(args, "bos_idx", 0)
        self.eos_idx = getattr(args, "eos_idx", None)

    @contextmanager
    def _temp_decode_cfg(self, sample_method: str, beam_size: Optional[int] = None, temperature: Optional[float] = None):
        """临时切换模型内部 args 的解码参数"""
        m = _unwrap_model(self.model)
        old_method = getattr(m.args, "sample_method", None)
        old_beam = getattr(m.args, "beam_size", None)
        old_temp = getattr(m.args, "temperature", None)

        if hasattr(m, "args"):
            m.args.sample_method = sample_method
            if beam_size is not None:
                m.args.beam_size = beam_size
            if temperature is not None:
                m.args.temperature = temperature

        try:
            yield
        finally:
            if hasattr(m, "args"):
                if old_method is not None:
                    m.args.sample_method = old_method
                if old_beam is not None:
                    m.args.beam_size = old_beam
                if old_temp is not None:
                    m.args.temperature = old_temp

    def _sample(self, images: torch.Tensor, method: str, temperature: Optional[float] = None):
        """
        返回: seq [B,T], seq_logprob_sum [B] or None
        """
        with self._temp_decode_cfg(method, beam_size=self.args.beam_size, temperature=temperature):
            seq, extra = self.model(images, mode='sample')

        seq_logprob_sum = None
        if isinstance(extra, torch.Tensor):
            # 常见情况1：extra 是每步选中 token 的 logprob: [B,T]
            if extra.dim() == 2 and extra.shape[:2] == seq.shape[:2]:
                seq_logprob_sum = extra.sum(dim=1)
            # 常见情况2：extra 是每步全词表 logprob/logits: [B,T,V]
            elif extra.dim() == 3 and extra.shape[0] == seq.shape[0] and extra.shape[1] == seq.shape[1]:
                logp = torch.log_softmax(extra, dim=-1)
                seq_logprob_sum = logp.gather(-1, seq.unsqueeze(-1)).squeeze(-1).sum(dim=1)

        return seq, seq_logprob_sum

    def _compute_seq_logprob_fallback(self, images: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:

        bsz, T = seq.size()
        bos = torch.full((bsz, 1), self.bos_idx, device=seq.device, dtype=seq.dtype)
        inp = torch.cat([bos, seq[:, :-1]], dim=1)  # [B, T]

        logits = self.model(images, inp, mode='train')  # [B, L, V]
        if logits.dim() != 3:
            raise RuntimeError("Unexpected logits shape: {}".format(tuple(logits.shape)))

        L = logits.size(1)
        target = seq[:, :L]
        logits = logits[:, :target.size(1), :]

        logp = torch.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [B,L]
        mask = build_eos_mask(target, self.eos_idx, self.pad_idx)
        return (logp * mask).sum(dim=1)

    def _train_epoch(self, epoch):
        self.logger.info('[{}/{}] RL fine-tuning on train set.'.format(epoch, self.epochs))
        self.model.train()

        total_loss = 0.0
        total_rl = 0.0
        total_ce = 0.0
        reward_sum = 0.0
        reward_base_sum = 0.0
        all_hit_sum = 0.0

        max_steps = self.args.max_steps_per_epoch
        steps = 0

        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
            if max_steps is not None and batch_idx >= max_steps:
                break

            images = images.to(self.device)
            reports_ids = reports_ids.to(self.device)
            reports_masks = reports_masks.to(self.device)

            # 1) 采样序列（policy）
            seq_s, seq_logp_sum = self._sample(images, method=self.args.sample_method_rl, temperature=self.args.temperature)
            if seq_logp_sum is None:
                seq_logp_sum = self._compute_seq_logprob_fallback(images, seq_s)

            # 2) baseline
            seq_b, _ = self._sample(images, method=self.args.baseline_method, temperature=None)

            # 3) reward
            reports_s = _unwrap_model(self.model).tokenizer.decode_batch(seq_s.detach().cpu().numpy())
            reports_b = _unwrap_model(self.model).tokenizer.decode_batch(seq_b.detach().cpu().numpy())

            r_s, all_s = compute_keyword_reward(reports_s, self.keyword_groups)
            r_b, _ = compute_keyword_reward(reports_b, self.keyword_groups)

            r_s_t = torch.tensor(r_s, device=self.device, dtype=torch.float32)
            r_b_t = torch.tensor(r_b, device=self.device, dtype=torch.float32)
            advantage = (r_s_t - r_b_t).detach()

            # 4) SCST loss
            rl_loss = -(advantage * seq_logp_sum).mean()

            # 5) 可选：CE
            ce_loss = torch.tensor(0.0, device=self.device)
            if self.ce_weight > 0:
                logits = self.model(images, reports_ids, mode='train')
                ce_loss = self.criterion(logits, reports_ids, reports_masks)

            loss = self.rl_weight * rl_loss + self.ce_weight * ce_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            steps += 1
            total_loss += loss.item()
            total_rl += rl_loss.item()
            total_ce += float(ce_loss.item()) if isinstance(ce_loss, torch.Tensor) else float(ce_loss)

            reward_sum += float(np.mean(r_s))
            reward_base_sum += float(np.mean(r_b))
            all_hit_sum += float(np.mean(all_s))

            if batch_idx % self.args.log_period == 0:
                self.logger.info(
                    '[{}/{}] Step {}/{}, loss={:.4f}, rl={:.4f}, ce={:.4f}, '
                    'reward={:.3f}, baseline={:.3f}, all_hit={:.3f}'.format(
                        epoch, self.epochs, batch_idx, len(self.train_dataloader),
                        total_loss / max(1, steps),
                        total_rl / max(1, steps),
                        total_ce / max(1, steps),
                        reward_sum / max(1, steps),
                        reward_base_sum / max(1, steps),
                        all_hit_sum / max(1, steps),
                    )
                )

        log = {
            'train_loss': total_loss / max(1, steps),
            'train_rl_loss': total_rl / max(1, steps),
            'train_ce_loss': total_ce / max(1, steps),
            'train_reward_avg': reward_sum / max(1, steps),
            'train_reward_baseline_avg': reward_base_sum / max(1, steps),
            'train_KEYWORD_ALL': all_hit_sum / max(1, steps),
        }

        # validation / test
        self.logger.info('[{}/{}] Evaluate on val set.'.format(epoch, self.epochs))
        val_met, val_kw = self._eval_split(self.val_dataloader, split_name="val")
        log.update(val_met); log.update(val_kw)

        self.logger.info('[{}/{}] Evaluate on test set.'.format(epoch, self.epochs))
        test_met, test_kw = self._eval_split(self.test_dataloader, split_name="test")
        log.update(test_met); log.update(test_kw)

        self.lr_scheduler.step()
        return log

    @torch.no_grad()
    def _eval_split(self, dataloader, split_name: str):
        self.model.eval()
        gts, res = [], []
        kw_reward_sum = 0.0
        kw_all_sum = 0.0
        n_batches = 0

        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(dataloader):
            images = images.to(self.device)
            reports_ids = reports_ids.to(self.device)

            seq, _ = self._sample(images, method=self.args.sample_method, temperature=None)
            reports = _unwrap_model(self.model).tokenizer.decode_batch(seq.cpu().numpy())
            ground_truths = _unwrap_model(self.model).tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

            res.extend(reports)
            gts.extend(ground_truths)

            r, all_r = compute_keyword_reward(reports, self.keyword_groups)
            kw_reward_sum += float(np.mean(r))
            kw_all_sum += float(np.mean(all_r))
            n_batches += 1

        met = self.metric_ftns({i: [gt] for i, gt in enumerate(gts)},
                               {i: [re] for i, re in enumerate(res)})
        met = {f'{split_name}_{k}': v for k, v in met.items()}

        kw = {
            f'{split_name}_KEYWORD_AVG': kw_reward_sum / max(1, n_batches),
            f'{split_name}_KEYWORD_ALL': kw_all_sum / max(1, n_batches),
        }
        return met, kw


# -------------------------
# 参数
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument('--image_dir', type=str, default=r'E:\images',
                        help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default=r"E:\1.json",
                        help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr', choices=['iu_xray', 'mimic_cxr'])
    parser.add_argument('--max_seq_length', type=int, default=60)
    parser.add_argument('--threshold', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)

    # Model settings
    parser.add_argument('--visual_extractor', type=str, default='vig_b_224')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True)

    # Transformer
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
    parser.add_argument('--mode', type=str, default='train')

    # for Cross-modal Memory
    parser.add_argument('--topk', type=int, default=32)
    parser.add_argument('--cmm_size', type=int, default=2048)
    parser.add_argument('--cmm_dim', type=int, default=512)

    # Sample related（评估/生成用）
    parser.add_argument('--sample_method', type=str, default='beam_search')
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sample_n', type=int, default=1)
    parser.add_argument('--group_size', type=int, default=1)
    parser.add_argument('--output_logsoftmax', type=int, default=1)
    parser.add_argument('--decoding_constraint', type=int, default=0)
    parser.add_argument('--block_trigrams', type=int, default=1)

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--save_dir', type=str, default=r'E:\hi')
    parser.add_argument('--record_dir', type=str, default=r'E:\hi')
    parser.add_argument('--log_period', type=int, default=200)
    parser.add_argument('--save_period', type=int, default=1)
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max', 'off'])
    # 建议监控 KEYWORD_ALL
    parser.add_argument('--monitor_metric', type=str, default='KEYWORD_ALL', help='KEYWORD_ALL or BLEU_4')
    parser.add_argument('--early_stop', type=int, default=50)

    # Optimization（RL 阶段建议更小 lr）
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr_ve', type=float, default=5e-6)
    parser.add_argument('--lr_ed', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.98))
    parser.add_argument('--adam_eps', type=float, default=1e-9)
    parser.add_argument('--amsgrad', type=bool, default=True)
    parser.add_argument('--noamopt_warmup', type=int, default=5000)
    parser.add_argument('--noamopt_factor', type=int, default=1)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR')
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.1)

    # Others
    parser.add_argument('--seed', type=int, default=9233)
    parser.add_argument('--resume', type=str, default=None,
                        help='resume full training (model+optimizer).')
    parser.add_argument('--pretrained', type=str, default=r'E:\model_best.pth',
                        help='load a pretrained checkpoint (ONLY model weights) before RL finetune')

    # -------- RL specific --------
    parser.add_argument('--sample_method_rl', type=str, default='sample',
                        help='RL 时用于采样的方式；一般用 sample（随机采样）')
    parser.add_argument('--baseline_method', type=str, default='greedy',
                        help='SCST baseline 解码方式；一般 greedy；不支持可用 beam_search 且 beam_size=1')
    parser.add_argument('--rl_weight', type=float, default=1.0)
    parser.add_argument('--ce_weight', type=float, default=0.05,
                        help='混一点 CE 防止模式崩坏；只想塞关键词可设为 0')
    parser.add_argument('--entropy_weight', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--max_steps_per_epoch', type=int, default=None)


    parser.add_argument('--keywords', type=str,
                        default='cardiomegaly,edema,pleural effusion,no cardiomegaly,no edema,no pleural effusion',
                        help='g1,g2,g3 with aliases using |.')

    return parser.parse_args()


def load_pretrained_weights(model: torch.nn.Module, ckpt_path: str, logger: logging.Logger):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded pretrained weights from {}".format(ckpt_path))
    logger.info("Missing keys: {}".format(len(missing)))
    logger.info("Unexpected keys: {}".format(len(unexpected)))


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    tokenizer = Tokenizer(args)

    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    model = models.models.BaseCMNModel(args, tokenizer)

    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    criterion = compute_loss
    metrics = modules.metrics.compute_scores

    keyword_groups = parse_keyword_groups(args.keywords)
    if len(keyword_groups) == 0:
        raise ValueError("No keywords parsed from --keywords")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger(__name__)

    if args.pretrained is not None:
        load_pretrained_weights(model, args.pretrained, logger)

    trainer = RLTrainer(
        model=model,
        criterion=criterion,
        metric_ftns=metrics,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        keyword_groups=keyword_groups,
    )
    trainer.train()


if __name__ == '__main__':
    main()
