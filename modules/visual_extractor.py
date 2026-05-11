import torch
import torch.nn as nn
from modules.vig import vig_b_224_gelu



class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = 'vig_b_224'
        self.pretrained = args.visual_extractor_pretrained
        self.grid_size = (7, 7)  #change

        self.vig = vig_b_224_gelu(pretrained=self.pretrained)

        # 冻结分类头
        for param in self.vig.prediction.parameters():
            param.requires_grad = False

        # 特征调整层
        self.feat_adjust = nn.Sequential(
            nn.Conv2d(640, args.d_model, 1),  # 输出通道与d_model一致
            nn.AdaptiveAvgPool2d((7, 7)))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, images):
        # ViG特征提取
        x = self.vig.stem(images) + self.vig.pos_embed
        x = self.vig.backbone(x)

        # 调整特征维度
        adjusted_feats = self.feat_adjust(x)  # (batch_size, d_model, 7, 7)

        # 生成网格坐标
        batch_size, _, feat_h, feat_w = adjusted_feats.shape

        grid_coords = self.generate_grid_coords(images.shape[-1], self.grid_size[0], self.grid_size[1]) #点图
        # 处理视觉特征
        patch_feats = adjusted_feats.flatten(2).permute(0, 2, 1)  # (batch_size, 49, d_model)
        avg_feats = self.avg_pool(adjusted_feats).view(batch_size, -1)  # (batch_size, d_model)

        return patch_feats, avg_feats, grid_coords

    @staticmethod
    def generate_grid_coords(img_size=224, grid_h=7, grid_w=7):
        stride_h = img_size // grid_h
        stride_w = img_size // grid_w
        coords = []
        for i in range(grid_h):
            for j in range(grid_w):
                x = j * stride_w + stride_w // 2
                y = i * stride_h + stride_h // 2
                coords.append([x, y])
        return torch.tensor(coords, dtype=torch.float32)
