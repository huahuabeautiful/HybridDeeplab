import os
import numpy as np
import torch
from PIL import Image
import time
from tqdm import tqdm
from deeplab import DeeplabV3
import warnings

warnings.filterwarnings('ignore')


class LargeImageSegmentor:
    def __init__(self, model_path=None, num_classes=2, backbone="xception", input_shape=[256, 256], cuda=True):
        """
        初始化大图像分割器
        Args:
            model_path: 模型权重路径
            num_classes: 类别数
            backbone: 骨干网络
            input_shape: 输入图像大小
            cuda: 是否使用GPU
        """
        # 初始化DeeplabV3模型
        self.deeplab = DeeplabV3(
            model_path=model_path,
            num_classes=num_classes,
            backbone=backbone,
            input_shape=input_shape,
            cuda=cuda
        )
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.cuda = cuda

    def split_image(self, image_path, patch_size=256, overlap=32):
        """
        将大图像分割成小块
        Args:
            image_path: 图像路径
            patch_size: 块大小
            overlap: 重叠像素（用于避免边界效应）
        Returns:
            patches: 图像块列表
            positions: 每个块在原图中的位置
            original_size: 原图尺寸
        """
        # 读取图像
        img = Image.open(image_path).convert('RGB')
        original_size = img.size  # (width, height)

        width, height = original_size
        patches = []
        positions = []

        # 计算步长
        step = patch_size - overlap

        # 使用进度条显示分割进度
        print(f"开始分割图像: {width}x{height} -> {patch_size}x{patch_size} patches")
        print(f"步长: {step}, 重叠: {overlap}")

        # 计算行数和列数
        cols = (width - overlap) // step + (1 if (width - overlap) % step > 0 else 0)
        rows = (height - overlap) // step + (1 if (height - overlap) % step > 0 else 0)

        print(f"将分割成 {rows} 行 × {cols} 列 = {rows * cols} 个块")

        # 遍历所有位置并裁剪图像块
        for y in range(0, height, step):
            for x in range(0, width, step):
                # 计算实际裁剪区域
                x_end = min(x + patch_size, width)
                y_end = min(y + patch_size, height)

                # 如果需要，调整起始位置以确保块大小一致
                x_start = max(0, x_end - patch_size)
                y_start = max(0, y_end - patch_size)

                # 裁剪图像块
                patch = img.crop((x_start, y_start, x_end, y_end))

                # 存储位置信息
                positions.append({
                    'x_start': x_start,
                    'y_start': y_start,
                    'x_end': x_end,
                    'y_end': y_end,
                    'original_position': (x, y, x_end - x_start, y_end - y_start)
                })
                patches.append(patch)

        return patches, positions, original_size

    def predict_patches(self, patches, batch_size=4):
        """
        批量预测图像块
        Args:
            patches: 图像块列表
            batch_size: 批处理大小
        Returns:
            predictions: 预测结果列表
        """
        predictions = []

        print(f"开始预测 {len(patches)} 个图像块...")

        # 分批处理以减少内存使用
        for i in tqdm(range(0, len(patches), batch_size), desc="预测进度"):
            batch_patches = patches[i:i + batch_size]
            batch_predictions = []

            for patch in batch_patches:
                # 使用get_miou_png方法获取分割结果（纯分割图，没有混合原图）
                # 你也可以使用detect_image方法，根据需要设置mix_type
                seg_result = self.deeplab.get_miou_png(patch)
                batch_predictions.append(seg_result)

            predictions.extend(batch_predictions)

            # 清理显存
            if self.cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return predictions

    def merge_predictions(self, predictions, positions, original_size, overlap_weight=True):
        """
        将预测结果拼接回原图大小
        Args:
            predictions: 预测结果列表
            positions: 位置信息列表
            original_size: 原图尺寸
            overlap_weight: 是否使用重叠区域加权平均
        Returns:
            merged_result: 拼接后的完整分割图
        """
        width, height = original_size
        num_classes = self.num_classes

        # 初始化结果数组
        if overlap_weight:
            # 使用加权平均需要两个数组：一个存储总和，一个存储权重
            result_sum = np.zeros((height, width, num_classes), dtype=np.float32)
            result_weight = np.zeros((height, width), dtype=np.float32)
        else:
            # 简单的覆盖方式
            result = np.zeros((height, width), dtype=np.uint8)

        print(f"开始拼接预测结果到 {width}x{height} 图像...")

        # 遍历所有预测块并放置到正确位置
        for idx, (pred, pos) in enumerate(tqdm(zip(predictions, positions),
                                               total=len(predictions),
                                               desc="拼接进度")):
            # 将预测结果转换为numpy数组
            pred_array = np.array(pred)  # (h, w)

            # 获取块的位置信息
            x_start, y_start = pos['x_start'], pos['y_start']
            x_end, y_end = pos['x_end'], pos['y_end']

            patch_height = y_end - y_start
            patch_width = x_end - x_start

            # 确保预测结果大小与块大小一致
            if pred_array.shape != (patch_height, patch_width):
                # 如果大小不匹配，调整预测结果大小
                pred_pil = Image.fromarray(pred_array)
                pred_pil = pred_pil.resize((patch_width, patch_height), Image.NEAREST)
                pred_array = np.array(pred_pil)

            if overlap_weight:
                # 为重叠区域创建权重图（中心区域权重高，边缘权重低）
                weight_map = self._create_weight_map(patch_height, patch_width)

                # 将预测结果转换为one-hot编码
                pred_onehot = np.eye(num_classes)[pred_array]

                # 累加到结果中
                result_sum[y_start:y_end, x_start:x_end] += pred_onehot * weight_map[:, :, np.newaxis]
                result_weight[y_start:y_end, x_start:x_end] += weight_map
            else:
                # 简单覆盖（适用于无重叠或少量重叠的情况）
                result[y_start:y_end, x_start:x_end] = pred_array

        if overlap_weight:
            # 计算加权平均
            result_weight[result_weight == 0] = 1  # 避免除以零
            result_prob = result_sum / result_weight[:, :, np.newaxis]
            result = np.argmax(result_prob, axis=2).astype(np.uint8)

        return Image.fromarray(result)

    def _create_weight_map(self, height, width, sigma=32):
        """
        创建权重图，用于重叠区域的加权平均
        """
        # 创建高斯权重图
        y, x = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, width), indexing='ij')
        d = np.sqrt(x * x + y * y)
        weight = np.exp(-(d ** 2) / (2 * sigma ** 2))

        # 归一化到0.5-1.0范围
        weight = 0.5 + 0.5 * (weight - weight.min()) / (weight.max() - weight.min())
        return weight

    def segment_large_image(self, image_path, output_path, patch_size=256, overlap=64, batch_size=4):
        """
        完整的大图像分割流程
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径
            patch_size: 块大小
            overlap: 重叠像素
            batch_size: 批处理大小
        """
        start_time = time.time()

        print("=" * 50)
        print(f"开始处理大图像: {image_path}")
        print("=" * 50)

        # 步骤1: 分割图像
        patches, positions, original_size = self.split_image(
            image_path, patch_size=patch_size, overlap=overlap
        )

        # 步骤2: 预测所有图像块
        predictions = self.predict_patches(patches, batch_size=batch_size)

        # 步骤3: 拼接预测结果
        merged_result = self.merge_predictions(
            predictions, positions, original_size, overlap_weight=True
        )

        # 步骤4: 保存结果
        merged_result.save(output_path)

        end_time = time.time()
        total_time = end_time - start_time

        print("=" * 50)
        print(f"处理完成!")
        print(f"输入图像: {image_path}")
        print(f"输出图像: {output_path}")
        print(f"原图尺寸: {original_size}")
        print(f"处理块数: {len(patches)}")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均每块: {total_time / len(patches):.3f}秒")
        print("=" * 50)

        return merged_result


# 使用示例
if __name__ == "__main__":
    # 初始化参数（根据你的训练配置修改）
    model_path = r'E:\Deeplabv3+\deeplabv3-plus.HybridASPP\logs\loss_2025_09_15_17_02_Xception.3102.Dice+CE.Unfreeze.HybridASPP\best_epoch_weights.pth'
    num_classes = 2
    backbone = "xception"

    # 初始化大图像分割器
    segmentor = LargeImageSegmentor(
        model_path=model_path,
        num_classes=num_classes,
        backbone=backbone,
        input_shape=[256, 256],
        cuda=True  # 如果有GPU则使用GPU
    )

    # 设置输入输出路径
    input_image_path = r"E:\Deeplabv3+\deeplabv3-plus.HybridASPP\keshan.png"  # 替换为你的大图像路径
    output_image_path = r"E:\Deeplabv3+\deeplabv3-plus.HybridASPP\generalization\segmentation_result.png"  # 输出路径

    # 执行分割
    # 参数说明：
    # - patch_size: 块大小，应与训练时相同（256）
    # - overlap: 重叠像素，建议设为32-64，可以减少边界伪影
    # - batch_size: 批处理大小，根据显存调整
    result = segmentor.segment_large_image(
        image_path=input_image_path,
        output_path=output_image_path,
        patch_size=256,
        overlap=64,
        batch_size=2  # 根据GPU显存调整
    )

    print("分割结果已保存至:", output_image_path)