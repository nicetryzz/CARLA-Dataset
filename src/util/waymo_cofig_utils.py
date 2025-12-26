import numpy as np
from numpy.linalg import inv
import os
from pathlib import Path
import pickle
import argparse
import imageio.v2 as imageio 
import json

class WaymoConverter:
    def __init__(self, root: str):
        """初始化转换器
        
        Args:
            root: 数据集根目录
        """
        # 设置路径
        self.root = Path(root)
        
        # 相机列表
        self.cameras = ['camera_FRONT', 'camera_FRONT_RIGHT', 'camera_BACK_RIGHT',
                       'camera_BACK', 'camera_BACK_LEFT', 'camera_FRONT_LEFT']

    def load_poses(self, path):
        """加载位姿数据
        
        Args:
            path: 位姿文件路径，例如 'root/poses.txt'
        
        Returns:
            np.ndarray: 形状为[N, 4, 4]的位姿矩阵数组，其中：
                - N 是帧数
                - 每个4x4矩阵表示一个变换矩阵 [R|t]
                    - R: 3x3旋转矩阵
                    - t: 3x1平移向量
        
        文件格式：
            - 每行包含12个数字（3x4矩阵展开）
            - 格式为：[r11 r12 r13 t1 r21 r22 r23 t2 r31 r32 r33 t3]
            - 函数会自动添加第4行 [0 0 0 1] 使其成为完整的4x4变换矩阵
        """
        # 加载数据
        data = np.loadtxt(path)  # [N, 12]
        
        # 重塑为[N, 3, 4]矩阵
        data = data.reshape(-1, 3, 4)  # [N, 3, 4]
        
        # 添加最后一行 [0, 0, 0, 1]
        new_rows = np.array([[0, 0, 0, 1]] * data.shape[0]).reshape(-1, 1, 4)  # [N, 1, 4]
        data = np.concatenate((data, new_rows), axis=1)  # [N, 4, 4]
        
        return data

    def load_sensor_params(self):
        """加载传感器参数（内参和外参）
        
        Args:
            camera_name: 相机名称，如 'camera_FRONT'
            
        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: 
                - extrinsic: [4, 4] 相机外参矩阵
                - intrinsic: [3, 3] 相机内参矩阵
                - distortion: [5] 相机畸变系数
        """
        # 从calib.txt文件中解析相机参数
        with open(self.root / "calib.txt", 'r') as f:
            lines = f.readlines()
        
        # 初始化所有相机的参数列表
        self.extrinsic = []
        self.intrinsic = []
        
        # 遍历每个相机
        for camera_id in range(len(self.cameras)):
            # 获取该相机的参数行
            rt_line = lines[camera_id * 2]
            k_line = lines[camera_id * 2 + 1]
            
            # 解析外参矩阵
            rt_values = [float(x) for x in rt_line.split(':')[1].split()]
            extr = np.array(rt_values).reshape(3, 4)
            extr = np.vstack((extr, [0, 0, 0, 1]))  # 添加最后一行
            self.extrinsic.append(extr)
            
            # 解析内参矩阵
            k_values = [float(x) for x in k_line.split(':')[1].split()]
            intr = np.array(k_values).reshape(3, 3)
            self.intrinsic.append(intr)
            
        # 转换为numpy数组
        self.extrinsic = np.array(self.extrinsic)  # [N, 4, 4]
        self.intrinsic = np.array(self.intrinsic)  # [N, 3, 3]
        
        # 从calib.txt文件中解析雷达外参
        tr_line = lines[-1]  
        tr_values = [float(x) for x in tr_line.split(':')[1].split()]
        self.lidar_extrinsic = np.array(tr_values).reshape(3, 4)
        self.lidar_extrinsic = np.vstack((self.lidar_extrinsic, [0, 0, 0, 1]))  # 添加最后一行

    def process_camera(self, camera_name):
        """处理单个相机数据"""
        img = imageio.imread(self.root / "images_depth" / f"{camera_name}" / "000000.png")
        
        camera_index = self.cameras.index(camera_name)
        extrinsic = self.extrinsic[camera_index]
        intrinsic = self.intrinsic[camera_index]
        distortion = np.zeros(5)

        # 计算相机到世界的变换
        c2w = self.car_pose @ extrinsic
        
        return {
            "hw": np.tile(np.array([img.shape[0],img.shape[1]]), (self.frame_num, 1)),
            "c2v": np.tile(extrinsic, (self.frame_num, 1, 1)).astype(np.float32),
            "sensor_v2w": self.car_pose,
            "c2w": c2w.astype(np.float32),
            "global_frame_ind": np.arange(self.frame_num),
            "intr": np.repeat(intrinsic[:3,:3].reshape(1,3,3), self.frame_num,  axis=0),
            "distortion": np.repeat(distortion.reshape(1,5), self.frame_num,  axis=0),
            "timestamp": self.timestamp  # NumPy数组格式的时间戳
        }

    def process_lidar(self):
        """处理激光雷达数据"""
        return {
            "l2v": np.tile(self.lidar_extrinsic, (self.frame_num, 1, 1)).astype(np.float32),
            "l2w": self.car_pose @ self.lidar_extrinsic,  # [N, 4, 4] 激光雷达位姿
            "global_frame_ind": np.arange(self.frame_num),
            "timestamp": self.timestamp  # NumPy数组格式的时间戳
        }

    def process_egocar(self):
        """处理自车数据"""   
        # 加载激光雷达时间戳作为自车时间戳,并转换为NumPy数组
        with open(self.root / "times.txt", 'r') as f:
            timestamp = f.readlines()
        self.timestamp = np.array([float(t.strip())/1e9 for t in timestamp], dtype=np.float64)  # 转换为NumPy数组
        
        return {
            "v2w": self.car_pose,  # [N, 4, 4] 自车位姿
            "global_frame_ind": np.arange(self.frame_num),
            "timestamp": self.timestamp  # NumPy数组格式的时间戳
        }

    def convert(self):
        """转换数据到Waymo格式"""
        # 加载位姿和标定数据
        self.car_pose = self.load_poses(self.root / "poses.txt")
        self.frame_num = self.car_pose.shape[0]
        self.load_sensor_params()
        
        # 构建数据字典
        data = {
            "scene_id": self.root.name,
            "metas": {
                "n_frames": self.frame_num
            },
            "observers": {},
            "objects": {}
        }
        
        # 添加自车
        data["observers"]["ego_car"] = {
            "class_name": "EgoVehicle",
            "n_frames": self.frame_num,
            "data": self.process_egocar(),
        }
        
        # 处理每个相机
        for cam_i, camera_name in enumerate(self.cameras):
            data["observers"][camera_name] = {
                "class_name": "Camera",
                "n_frames": self.frame_num,
                "data": self.process_camera(camera_name),
            }
        
        # 添加激光雷达
        data["observers"]["lidar_TOP"] = {
            "class_name": "RaysLidar",
            "n_frames": self.frame_num,
            "data": self.process_lidar(),
        }
        
        # 保存为.pt文件
        output_path = self.root / "scenario.pt"
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"数据已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert dataset to Waymo format')
    parser.add_argument('--base_dir', type=str, 
                    default="/home/hqlab/workspace/dataset/parkinglot",
                    help='Base directory')
    parser.add_argument('--subdir', type=str, 
                    default="data/10_26",
                    help='Subdirectory containing data')
    args = parser.parse_args()
    
    # 创建转换器并执行转换
    converter = WaymoConverter(
        base_dir=args.base_dir,
        subdir=args.subdir
    )
    converter.convert()

if __name__ == "__main__":
    main()