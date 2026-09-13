# magic_eye_node.py
# Magic Eye Stereogram Generator for ComfyUI

import torch
import numpy as np
from PIL import Image

class MagicEyeGenerator:
    """
    魔法全息立体图生成器
    纹理单张，深度图支持单张/批量
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "纹理图片": ("IMAGE", {
                    "tooltip": "纹理图片：单张即可，将应用到所有深度图"
                }),
                "深度图片": ("IMAGE", {
                    "tooltip": "深度图：支持单张或批量输入（亮色=凸起，暗色=凹陷）"
                }),
                "深度强度": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "控制立体感的强弱，值越大立体感越强"
                }),
                "纹理宽度": ("INT", {
                    "default": 100,
                    "min": 50,
                    "max": 200,
                    "step": 5,
                    "display": "slider",
                    "tooltip": "纹理重复的宽度，影响立体图效果"
                }),
                "输出宽度": ("INT", {
                    "default": 800,
                    "min": 200,
                    "max": 1920,
                    "step": 10,
                    "tooltip": "生成图片的宽度（像素）"
                }),
                "输出高度": ("INT", {
                    "default": 600,
                    "min": 200,
                    "max": 1080,
                    "step": 10,
                    "tooltip": "生成图片的高度（像素）"
                }),
                "对比度增强": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.5,
                    "max": 2.5,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "深度图的对比度增强，值越大深度层次越分明"
                }),
            },
            "optional": {
                "随机种子": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "随机种子，控制纹理采样的随机性"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("立体图",)
    FUNCTION = "generate"
    CATEGORY = "图像/生成"
    OUTPUT_NODE = False
    
    def generate(self, 纹理图片, 深度图片, 深度强度, 纹理宽度,
                 输出宽度, 输出高度, 对比度增强, 随机种子=0):
        """
        生成立体图
        """
        np.random.seed(随机种子)
        
        # 转换为 numpy
        texture_np = 纹理图片.cpu().numpy()
        depth_np = 深度图片.cpu().numpy()
        
        # 纹理图：只取第一张
        tex = texture_np[0]
        
        # 深度图批次
        depth_batch = depth_np
        batch_size = depth_batch.shape[0]
        
        # 预处理纹理（只需要做一次）
        texture_tiled = self._prepare_texture(tex, 输出宽度, 输出高度, 纹理宽度)
        
        results = []
        
        # 批量处理深度图
        for i in range(batch_size):
            dep = depth_batch[i]
            
            # 处理深度图
            depth_processed = self._process_depth(dep, 输出宽度, 输出高度, 对比度增强)
            
            # 生成立体图
            stereogram = self._generate_stereogram(
                texture_tiled, depth_processed, 深度强度, 纹理宽度
            )
            
            results.append(stereogram)
        
        # 转换为 tensor
        result_tensor = torch.from_numpy(np.array(results)).float()
        return (result_tensor,)
    
    def _process_depth(self, depth, target_w, target_h, contrast_boost):
        """处理深度图：灰度化 + 归一化 + 调整尺寸 + 对比度增强"""
        # 转换为灰度
        if len(depth.shape) == 3 and depth.shape[-1] == 3:
            gray = np.mean(depth, axis=-1)
        elif len(depth.shape) == 3 and depth.shape[-1] == 1:
            gray = depth.squeeze(-1)
        elif len(depth.shape) == 2:
            gray = depth
        else:
            gray = depth.squeeze()
        
        # 转为 PIL 调整尺寸
        gray_normalized = np.clip(gray, 0, 1)
        img_pil = Image.fromarray((gray_normalized * 255).astype(np.uint8))
        img_resized = img_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
        result = np.array(img_resized).astype(np.float32) / 255.0
        
        # 归一化
        min_val = result.min()
        max_val = result.max()
        if max_val > min_val:
            result = (result - min_val) / (max_val - min_val)
        
        # 对比度增强
        result = np.clip((result - 0.5) * contrast_boost + 0.5, 0, 1)
        
        return result
    
    def _prepare_texture(self, texture, target_w, target_h, pattern_width):
        """准备纹理：保持比例平铺到目标尺寸"""
        h, w, c = texture.shape
        
        # 如果是单通道，扩展为3通道
        if c == 1:
            texture = np.repeat(texture, 3, axis=-1)
        
        # 限制数值范围
        texture = np.clip(texture, 0, 1)
        
        # 转换为 PIL
        img_pil = Image.fromarray((texture * 255).astype(np.uint8))
        
        # 计算纹理的平铺尺寸
        aspect_ratio = h / w
        tile_height = int(pattern_width * aspect_ratio)
        
        # 调整平铺尺寸
        if tile_height < 1:
            tile_height = 1
        if tile_height > target_h:
            tile_height = target_h // 2
            if tile_height < 1:
                tile_height = 1
            tile_width = int(tile_height / aspect_ratio)
            if tile_width < 1:
                tile_width = 1
        else:
            tile_width = pattern_width
        
        # 缩放纹理到平铺尺寸
        texture_resized = img_pil.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        texture_array = np.array(texture_resized).astype(np.float32) / 255.0
        
        # 平铺纹理到目标尺寸
        result = np.zeros((target_h, target_w, 3), dtype=np.float32)
        
        for y in range(0, target_h, tile_height):
            for x in range(0, target_w, tile_width):
                block_h = min(tile_height, target_h - y)
                block_w = min(tile_width, target_w - x)
                
                if block_h < tile_height or block_w < tile_width:
                    result[y:y+block_h, x:x+block_w] = texture_array[:block_h, :block_w]
                else:
                    result[y:y+tile_height, x:x+tile_width] = texture_array
        
        return result
    
    def _generate_stereogram(self, texture, depth, depth_strength, pattern_width):
        """核心立体图生成算法"""
        h, w, c = texture.shape
        result = np.zeros((h, w, c), dtype=np.float32)
        
        # 最大偏移量
        max_shift = int(pattern_width * 0.4)
        min_shift = 1
        
        # 预计算深度偏移映射
        depth_shift = (depth * depth_strength * max_shift).astype(np.int32)
        depth_shift = np.clip(depth_shift, min_shift, max_shift)
        
        # 逐行生成
        for y in range(h):
            row_buffer = np.zeros((w, c))
            
            # 初始化：使用纹理填充前 pattern_width 个像素
            for x in range(min(pattern_width, w)):
                row_buffer[x] = texture[y, x % w]
            
            # 生成剩余像素
            for x in range(pattern_width, w):
                shift = depth_shift[y, x]
                source_x = x - pattern_width + shift
                
                if 0 <= source_x < x:
                    row_buffer[x] = row_buffer[source_x]
                else:
                    row_buffer[x] = texture[y, x % w]
            
            result[y] = row_buffer
        
        return result


# 注册节点
NODE_CLASS_MAPPINGS = {
    "MagicEyeGenerator": MagicEyeGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MagicEyeGenerator": "🎨 魔法立体图生成器",
}