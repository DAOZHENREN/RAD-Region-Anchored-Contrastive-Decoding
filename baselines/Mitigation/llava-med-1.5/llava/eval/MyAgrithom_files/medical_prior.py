import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import densenet121
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt


CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

class CheXNetPriorProvider:
    def __init__(self, weight_path='chexnet_weights.pth', device='cuda:0'):
        """
        初始化：加载模型并提取出最后一层线性分类器的权重。
        """
        self.device = device
        print(f"[*] 正在初始化极速版 CheXNet 先验模块至 {self.device}...")
        
        # 1. 定义网络
        self.model = densenet121(pretrained=False)
        num_ftrs = self.model.classifier.in_features  # DenseNet121 是 1024 维特征
        
        # 保持与 CheXNet 结构一致
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 14),
            torch.nn.Sigmoid()
        )
        
        # 2. 加载权重
        checkpoint = torch.load(weight_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.model.to(self.device)
        self.model.eval() # 开启评估模式
        
        # ==========================================================
        # 🌟 核心工程技巧：把分类器的权重单独“抠”出来缓存！
        # 这个权重矩阵形状是 [14, 1024]，代表 14 种疾病对 1024 个特征通道的关注度
        # ==========================================================
        self.fc_weights = self.model.classifier[0].weight.data.clone().to(self.device)
        
        # =========================================================
        # 🌟 新增：专门为 CheXNet 准备的图像预处理管道
        # =========================================================
        self.transform = T.Compose([
            T.Resize((224, 224), antialias=True), # 强行缩放到 CheXNet 需要的尺寸
            T.ToTensor(),                         # 转为 Tensor，并将像素值缩放至[0, 1]
            T.Normalize(                          # ImageNet 标准归一化（CheXNet 训练必须项）
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        print("[*] 极速版模块加载完毕！")

    def get_all_14_heatmaps(self, raw_image, llava_grid_size=24):
        """
        纯前向传播，时间复杂度 O(1)，一次性返回 14 张热力图！
        输入: input_tensor[1, 3, 224, 224] (你的X光片)
        输出: tensor 形状为 [14, 24, 24]，包含了 14 种病灶的空间热力图
        """
        # 🌟 关键防御：医学 X 光片通常是单通道灰度图 ('L' 模式)，
        # 而预处理管道需要 3 通道 ('RGB' 模式)，必须先转换！
        if raw_image.mode != 'RGB':
            raw_image = raw_image.convert('RGB')
        
        # 1. 直接对最原始的 PIL 图像应用标准转换
        clean_tensor = self.transform(raw_image) # 变成形状 [3, 224, 224]

         # 2. 增加 Batch 维度并送入显卡
        clean_tensor = clean_tensor.unsqueeze(0).to(self.device) # 变成[1, 3, 224, 224]
        
        with torch.no_grad(): # 绝对不计算梯度，节省大量显存和时间！
            # 1. 执行一次前向传播，只运行卷积层，拿到空间特征图
            # features 形状:[1, 1024, 7, 7] (7x7是 DenseNet 输出的空间分辨率)
            features = self.model.features(clean_tensor)
            features = F.relu(features, inplace=True)
            
        # 2. 🌟 矩阵乘法魔法 (Einsum) 🌟
        # 将分类权重 [14, 1024] 与特征图[1, 1024, 7, 7] 相乘
        # 这一步数学含义：直接算出 14 种疾病在 7x7 网格上的激活程度！
        cam = torch.einsum('nc,bchw->bnhw', self.fc_weights, features) # 输出[1, 14, 7, 7]
        
        # 消除负数激活（只关注有正向促进作用的特征）
        cam = F.relu(cam)
        
        # 3. 将 7x7 的热力图插值放大到 LLaVA 的 24x24 尺寸
        aligned_cam = F.interpolate(
            cam, 
            size=(llava_grid_size, llava_grid_size), 
            mode='bilinear', 
            align_corners=False
        ) # 输出[1, 14, 24, 24]

        # 去掉 batch 维度，返回[14, 24, 24] 的张量
        return aligned_cam.squeeze(0)

    def get_all_topk_indices(self, all_heatmaps, boost_num=5):
        """
        一次性返回 14 种疾病的 Top-K 索引，包含边缘斩杀过滤。
        输入: all_heatmaps [14, 24, 24]
        输出: List[List[int]]，长度为 14，每个子列表包含 boost_num 个索引
        """
        # clone 一份，防止修改原始的热力图张量
        heatmaps = all_heatmaps.clone() 
        
        # =========================================================
        # 🛡️ 核心修复：无情斩杀 14 张图的边缘伪影！
        # 强行把最上面 2 行、最下面 2 行、最左侧 2 列、最右侧 2 列归零！
        # 彻底干掉黑框、PORTABLE 字母和 CNN 零填充白边的干扰
        # =========================================================
        heatmaps[:, 0:2, :] = 0.0   # 顶端 2 行归零
        heatmaps[:, -2:, :] = 0.0   # 底端 2 行归零
        heatmaps[:, :, 0:2] = 0.0   # 左侧 2 列归零
        heatmaps[:, :, -2:] = 0.0   # 右侧 2 列归零

        # 展平为 [14, 576] 维张量
        flat_heatmaps = heatmaps.view(14, -1)

        # 极速取 Top-K (沿着维度 1，也就是在这 576 个坐标里挑)
        topk_values, topk_indices = torch.topk(flat_heatmaps, boost_num, dim=1)
        filtered_topk_list = []
        for i in range(14): # 遍历 14 种疾病
            # 生成布尔掩码：只有热力值严格大于 0 的才被认为是有效的
            # (考虑到浮点数精度，也可以用 topk_values[i] > 1e-5)
            valid_mask = topk_values[i] > 1e-5
            
            # 使用掩码筛选出有效的索引，并转换为普通的 Python 列表
            valid_indices = topk_indices[i][valid_mask].tolist()
            filtered_topk_list.append(valid_indices)
        return filtered_topk_list

    def __del__(self):
        if hasattr(self, 'model'): del self.model
        if hasattr(self, 'fc_weights'): del self.fc_weights
        try: torch.cuda.empty_cache()
        except: pass

def visualize_and_save_14_heatmaps(raw_image: Image.Image, all_heatmaps: torch.Tensor, save_path="med_opera_heatmaps.png"):
    """
    生成 14 张融合了原图的伪彩色热力图，并拼成一张大图保存。
    
    参数:
        raw_image: 原始的 X 光片 (PIL.Image 对象)
        all_heatmaps: 从 prior_provider 提取出的张量，形状为 [14, 24, 24]
        save_path: 大图保存的路径
    """
    print("[*] 正在生成 14 种疾病的融合热力图可视化...")
    
    # 14 种标准疾病名称
    chexnet_classes =[
        "Atelectasis (lung)", "Cardiomegaly (heart)", "Effusion (jiye)", "Infiltration (jingrun)", "Mass (zhongkuai)", 
        "Nodule (jiejie)", "Pneumonia (lung)", "Pneumothorax (qixiong)", "Consolidation (shibiang)", "Edema (shuizhong)", 
        "Emphysema (feijqizhong)", "Fibrosis (qiangweihua)", "Pleural Thickening (xiongmozenghou)", "Hernia (shangqi)"
    ]

    # 1. 准备原图 (转为 numpy 数组)
    if raw_image.mode != 'RGB':
        raw_image = raw_image.convert('RGB')
    img_arr = np.array(raw_image)
    
    # 获取原图的宽高，供后面放大热力图使用
    img_height, img_width = img_arr.shape[0], img_arr.shape[1]
    
    # 将原图归一化到 [0, 1] 方便后续透明度融合
    img_arr_float = img_arr.astype(np.float32) / 255.0

    # 2. 设置 Matplotlib 画布 (3 行 5 列，共 15 个格子：1 个原图 + 14 个热力图)
    fig, axes = plt.subplots(3, 5, figsize=(24, 14))
    axes = axes.flatten()

    # 在第一个格子画原图
    axes[0].imshow(img_arr)
    axes[0].set_title("Original X-Ray", fontweight='bold', fontsize=14)
    axes[0].axis('off')
    
    # 获取 Matplotlib 的 jet colormap
    jet_cmap = plt.get_cmap('jet')

    # 3. 遍历 14 张热力图
    for i in range(14):
        # 取出单张 24x24 的热力图并转为 numpy
        heatmap_24 = all_heatmaps[i].detach().cpu().float().numpy()

        # =========================================================
        # 🛡️ 可选：如果你想看到“边缘斩杀”后的真实效果，取消下面 4 行注释
        # heatmap_24[0:2, :] = 0.0
        # heatmap_24[-2:, :] = 0.0
        # heatmap_24[:, 0:2] = 0.0
        # heatmap_24[:, -2:] = 0.0
        # =========================================================

        # a) 归一化热力图到 [0, 1]
        max_val = heatmap_24.max()
        min_val = heatmap_24.min()
        if max_val > min_val:
            heatmap_normalized = (heatmap_24 - min_val) / (max_val - min_val)
        else:
            heatmap_normalized = heatmap_24  # 全黑的情况

        # b) 放大热力图 (Resize) - 使用 PIL 替代 cv2
        # 将归一化后的 numpy 数组转为 PIL Image 进行双三次插值放大
        heatmap_pil = Image.fromarray(heatmap_normalized)
        
        # 兼容不同版本的 PIL，建议使用 Image.Resampling.BICUBIC，老版本用 Image.BICUBIC
        try:
            resample_filter = Image.Resampling.BICUBIC
        except AttributeError:
            resample_filter = Image.BICUBIC
            
        heatmap_resized_pil = heatmap_pil.resize((img_width, img_height), resample=resample_filter)
        heatmap_resized = np.array(heatmap_resized_pil)

        # c) 应用 JET 伪彩色映射 - 使用 matplotlib 替代 cv2
        # jet_cmap 接收 [0, 1] 的数据，直接返回形状为 (H, W, 4) 的 RGBA 浮点数组 (范围 0.0-1.0)
        colormap_rgba = jet_cmap(heatmap_resized)
        
        # 我们只需要前三个通道 (RGB) 进行融合
        colormap_float = colormap_rgba[..., :3].astype(np.float32)

        # d) ✨ 图像融合 (Alpha Blending) ✨
        # 公式: 0.6 * 原图 + 0.4 * 热力图 (你可以自己调这个比例，比如 0.5:0.5)
        blended = 0.6 * img_arr_float + 0.4 * colormap_float
        blended = np.clip(blended, 0, 1)

        # e) 绘制到子图上
        ax = axes[i + 1]
        ax.imshow(blended)
        ax.set_title(chexnet_classes[i], fontsize=12)
        ax.axis('off')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 4. 调整排版并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # dpi=300 保证论文出版级的高清画质
    
    # 显式关闭画布释放内存
    plt.close(fig)
    print(f"[*] ✅ 热力图生成完毕！已保存至: {save_path}")
    
