import os
import shutil
from pathlib import Path

def collect_mask_files(dataset_root, output_dir="all_masks"):
    """
    将 dataset_root 下所有子文件夹中的 mask_obj1.png 复制到 output_dir。

    Args:
        dataset_root (str): 数据集根目录路径（包含 00473, 00474 等子文件夹）
        output_dir (str): 目标文件夹名称（默认 'all_masks'）
    """
    # 创建输出目录（如果不存在）
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 遍历 dataset_root 下的所有子目录
    for entry in os.listdir(dataset_root):
        sub_dir = Path(dataset_root) / entry
        if not sub_dir.is_dir():
            continue   # 跳过非文件夹

        src_file = sub_dir / "mask_obj1.png"
        if not src_file.exists():
            print(f"警告: {src_file} 不存在，跳过")
            continue

        # 构造目标文件名：加上子目录名前缀以避免重名
        # 例如 00473_mask_obj1.png
        dst_filename = f"{entry}_mask_obj1.png"
        dst_path = out_path / dst_filename

        # 复制文件（保留元数据）
        try:
            shutil.copy2(src_file, dst_path)
            print(f"已复制: {src_file} -> {dst_path}")
        except Exception as e:
            print(f"复制 {src_file} 失败: {e}")

if __name__ == "__main__":
    # 请将 'dataset_root' 替换为你的数据集实际根目录路径
    dataset_root = "/home/tenda/data/serve_bread/realsense/rs_serve_bread_000/preprocess/all_data"   # 示例路径，请修改
    collect_mask_files(dataset_root)