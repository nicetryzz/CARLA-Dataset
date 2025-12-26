import numpy as np
import os
import sqlite3
import collections
import quaternion
from typing import List, Dict

class ColmapConvertor:
    def __init__(self):
        self.start_frame = 0
        self.end_frame = 0
        self.sampling_rate = 1.0
        self.image_list = {}  # 用字典存储每个相机的图像列表
        self.camera_list = [
            'camera_FRONT', 'camera_FRONT_LEFT', 'camera_FRONT_RIGHT',
            'camera_BACK', 'camera_BACK_LEFT', 'camera_BACK_RIGHT'
        ]
        self.expose_list = ['expose_0', 'expose_1', 'expose_2', 'expose_3', 'expose_4']
        
    def set_frame_range(self, start_frame: int, end_frame: int):
        """设置要处理的帧范围
        
        Args:
            start_frame (int): 起始帧
            end_frame (int): 结束帧
        """
        self.start_frame = start_frame
        self.end_frame = end_frame
        
    def set_sampling_rate(self, rate: float):
        """设置采样率
        
        Args:
            rate (float): 采样率(Hz)
        """
        self.sampling_rate = rate
        
    def sample_images(self, root_dir: str, expose_ids: List[int] = None) -> Dict[str, Dict[str, Dict]]:
        """根据设定的参数采样图像和对应的参数
        
        Args:
            root_dir (str): 根目录路径=
            expose_ids (List[int], optional): 需要处理的曝光ID列表，默认处理所有曝光
            
        Returns:
            Dict[str, Dict[str, Dict]]: 采样后的图像路径和参数字典
            格式为: {
                'camera_name': {
                    'expose_id': {
                        'image_paths': [image_paths],
                        'extrinsics': [4x4_matrices],
                        'intrinsic': [3x3_matrix],
                        'distortion': [5_coefficients]
                    }
                }
            }
        """
        if not os.path.exists(root_dir):
            raise ValueError(f"根目录 {root_dir} 不存在")
            
        # 加载scenario.pt中的参数数据
        import pickle
        scenario_path = os.path.join(root_dir, 'scenario.pt')
        with open(scenario_path, 'rb') as f:
            scenario_data = pickle.load(f)
            
        # 如果没有指定曝光ID，则处理所有曝光
        if expose_ids is None:
            expose_ids = list(range(5))  # 0-4
        
        result = {}
        
        # 遍历每个相机
        img_path = os.path.join(root_dir, 'images')
        for camera in self.camera_list:
            camera_path = os.path.join(img_path, camera)
            if not os.path.exists(camera_path):
                continue
                
            result[camera] = {}
            
            # 获取该相机的所有参数
            if camera in scenario_data['observers']:
                camera_data = scenario_data['observers'][camera]['data']
                camera_extrinsics = camera_data['c2w']
                camera_intrinsic = camera_data['intr'][0]  # 取第一帧的内参
                camera_distortion = camera_data['distortion'][0]  # 取第一帧的畸变系数
            else:
                continue
            
            # 遍历指定的曝光
            for expose_id in expose_ids:
                expose_name = f'expose_{expose_id}'
                expose_path = os.path.join(camera_path, expose_name)
                
                if not os.path.exists(expose_path):
                    continue
                    
                # 获取该曝光下的所有图像
                images = sorted([f for f in os.listdir(expose_path) 
                               if f.endswith('.png')])
                
                # 根据起始帧和结束帧截取
                valid_images = images[self.start_frame:self.end_frame+1]
                valid_extrinsics = camera_extrinsics[self.start_frame:self.end_frame+1]
                
                # 按照采样率采样
                sampled_images = valid_images[::self.sampling_rate]
                sampled_extrinsics = valid_extrinsics[::self.sampling_rate]
                
                # 存储完整路径和对应的参数
                result[camera][expose_name] = {
                    'image_paths': [os.path.join(expose_path, img) for img in sampled_images],
                    'extrinsics': sampled_extrinsics,
                    'intrinsic': camera_intrinsic,
                    'distortion': camera_distortion
                }
                
        return result

    def write_colmap_format(self, camera_data: Dict, output_dir: str):
        """将相机参数写入COLMAP格式的txt文件
        
        Args:
            camera_data (Dict): sample_images返回的相机数据
            output_dir (str): 输出目录路径
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 写入cameras.txt
        with open(os.path.join(output_dir, 'cameras.txt'), 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            
            camera_id = 1
            for camera_name, expose_dict in camera_data.items():
                for expose_name, data in expose_dict.items():
                    # 获取图像尺寸
                    img_path = data['image_paths'][0]
                    from PIL import Image
                    with Image.open(img_path) as img:
                        width, height = img.size
                    
                    # 提取内参
                    K = data['intrinsic']
                    fx = K[0, 0]
                    fy = K[1, 1]  # 现在分别使用fx和fy
                    cx, cy = K[0, 2], K[1, 2]
                    
                    # PINHOLE相机模型参数: fx, fy, cx, cy
                    params = [fx, fy, cx, cy]
                    params_str = ' '.join(map(str, params))
                    
                    f.write(f"{camera_id} PINHOLE {width} {height} {params_str}\n")
                    camera_id += 1
        
        # 写入images.txt
        with open(os.path.join(output_dir, 'images.txt'), 'w') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            
            image_id = 1
            camera_id = 1
            for camera_name, expose_dict in camera_data.items():
                for expose_name, data in expose_dict.items():
                    for image_path, extrinsic in zip(data['image_paths'], data['extrinsics']):
                        # 从外参矩阵中提取旋转矩阵和平移向量
                        R = extrinsic[:3, :3]
                        t = extrinsic[:3, 3]
                        
                        # COLMAP使用世界到相机的变换，需要求逆
                        R_inv = R.T
                        t_inv = -R.T @ t
                        
                        # 将旋转矩阵转换为四元数
                        import quaternion
                        q = quaternion.from_rotation_matrix(R_inv)
                        qw, qx, qy, qz = q.w, q.x, q.y, q.z
                        
                        # 写入图像信息
                        f.write(f"{image_id} {qw} {qx} {qy} {qz} {t_inv[0]} {t_inv[1]} {t_inv[2]} {camera_id} {os.path.basename(image_path)}\n")
                        # 空行（因为没有特征点信息）
                        f.write("\n")
                        
                        image_id += 1
                    camera_id += 1
        
        # 写入空的points3D.txt
        with open(os.path.join(output_dir, 'points3D.txt'), 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

    def prepare_colmap_workspace(self, base_dir: str, root_dir: str, expose_ids: List[int] = None):
        """准备COLMAP工作空间，组织图像和参数文件
        
        Args:
            base_dir (str): COLMAP工作空间根目录
            root_dir (str): 原始图像根目录
            scenario_path (str): scenario.pt文件路径
            expose_ids (List[int], optional): 需要处理的曝光ID列表
        """
        # 创建目录结构
        images_dir = os.path.join(base_dir, 'images')
        sparse_dir = os.path.join(base_dir, 'sparse')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(sparse_dir, exist_ok=True)
        
        # 采样图像和参数
        camera_data = self.sample_images(root_dir, expose_ids)
        
        # 复制图像到images目录
        for camera_name, expose_dict in camera_data.items():
            for expose_name, data in expose_dict.items():
                for src_path in data['image_paths']:
                    # 复制图像文件
                    dst_name = f"{camera_name}_{expose_name}_{os.path.basename(src_path)}"
                    dst_path = os.path.join(images_dir, dst_name)
                    import shutil
                    shutil.copy2(src_path, dst_path)
                    
                    # 更新camera_data中的图像路径
                    idx = data['image_paths'].index(src_path)
                    data['image_paths'][idx] = dst_path
        
        # 写入COLMAP格式的文件到sparse目录
        self.write_colmap_format(camera_data, sparse_dir)
        
        print(f"COLMAP工作空间已准备完成：")
        print(f"- 图像目录：{images_dir}")
        print(f"- 稀疏重建目录：{sparse_dir}")

def main():
    # 测试参数
    base_dir = "/home/hqlab/workspace/reconstruction/result/final_result/carla/scene_0/colmap"  # COLMAP工作空间目录
    root_dir = "/home/hqlab/workspace/dataset/carla_data/dumper/2024_12_29_13_59_40"  # 原始图像目录
    
    # 创建转换器实例
    convertor = ColmapConvertor()
    
    # 设置参数
    convertor.set_frame_range(30, 160)  # 处理前100帧
    convertor.set_sampling_rate(2)   # 2Hz采样率
    
    # 准备COLMAP工作空间
    convertor.prepare_colmap_workspace(
        base_dir=base_dir,
        root_dir=root_dir,
        expose_ids=[2]  # 只处理expose_2
    )

if __name__ == "__main__":
    main()

