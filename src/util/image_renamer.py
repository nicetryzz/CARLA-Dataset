import os
import shutil
from pathlib import Path
from PIL import Image

def rename_and_copy_images(src_dir: str, dst_dir: str, extension: str = 'jpg'):
    """将源文件夹中的图片按序列重命名并复制到目标文件夹
    
    Args:
        src_dir (str): 源文件夹路径
        dst_dir (str): 目标文件夹路径
        extension (str, optional): 目标图片格式，默认为'jpg'
    """
    # 创建目标文件夹
    os.makedirs(dst_dir, exist_ok=True)
    
    # 获取所有图片文件
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = []
    for file in os.listdir(src_dir):
        if file.lower().endswith(valid_extensions):
            image_files.append(file)
    
    # 按文件名排序
    image_files.sort()
    
    # 重命名并复制
    for idx, old_name in enumerate(image_files):
        # 生成新文件名 (00000.jpg, 00001.jpg, ...)
        new_name = f"{idx:05d}.{extension}"
        
        src_path = os.path.join(src_dir, old_name)
        dst_path = os.path.join(dst_dir, new_name)
        
        # 如果需要转换格式
        if not old_name.lower().endswith(f'.{extension}'):
            img = Image.open(src_path)
            img.save(dst_path)
        else:
            shutil.copy2(src_path, dst_path)
            
        print(f"处理: {old_name} -> {new_name}")
    
    print(f"\n共处理了 {len(image_files)} 张图片")
    print(f"源文件夹: {src_dir}")
    print(f"目标文件夹: {dst_dir}")

def main():
    # 示例用法
    src_dir = "/home/hqlab/workspace/reconstruction/result/final_result/carla/scene_0/colmap/images"  # 替换为源文件夹路径
    dst_dir = "/home/hqlab/workspace/reconstruction/result/final_result/carla/scene_0/colmap/images_change"  # 替换为目标文件夹路径
    
    rename_and_copy_images(
        src_dir=src_dir,
        dst_dir=dst_dir,
        extension='jpg'  # 可以指定目标格式
    )

if __name__ == "__main__":
    main() 