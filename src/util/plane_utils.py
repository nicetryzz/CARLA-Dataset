import numpy as np
import cv2
import matplotlib.pyplot as plt

def filter_by_color(image_path, target_color, threshold=0):
    """根据目标颜色过滤图像并可视化
    
    Args:
        image_path: 图像路径
        target_color: 目标颜色RGB值,如[255,0,0]表示红色
        threshold: 颜色匹配的阈值,默认0表示完全匹配
        
    Returns:
        mask: 过滤后的二值图像
    """
    # 读取图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print(img.max())
    
    # 转换为numpy数组
    target_color = np.array(target_color)
    
    # 创建掩码,只保留完全匹配的像素
    mask = np.all(img == target_color, axis=2)
    
    # 可视化
    plt.figure(figsize=(12,4))
    
    plt.subplot(131)
    plt.imshow(img)
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(132) 
    plt.imshow(mask, cmap='gray')
    plt.title('过滤掩码')
    plt.axis('off')
    
    plt.subplot(133)
    filtered = img.copy()
    filtered[~mask] = 0
    plt.imshow(filtered)
    plt.title('过滤结果')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return mask


def main():
    image_path = '/home/hqlab/workspace/CARLA-Dataset/temp/2024_12_26_17_09_40/images_semantic/camera_BACK/000000.png'
    target_color = [29,0,0]
    filter_by_color(image_path, target_color)

if __name__ == '__main__':
    main()