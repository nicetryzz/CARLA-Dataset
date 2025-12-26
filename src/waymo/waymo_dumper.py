import os
import cv2
import numpy as np
import open3d as o3d
import copy
from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager
from packages.carla1s.actors import Vehicle, RgbCamera, DepthCamera, SimpleLidar, SemanticCamera
from src.util import *
import imageio.v2 as imageio
import sys
from pathlib import Path
sys.path.append("/home/hqlab/workspace/CARLA-Dataset")  # 将工作区根目录添加到Python路径

from packages.carla1s.actors import Sensor
from packages.carla1s.tf import Point, CoordConverter, Transform, Coordinate


from packages.carla1s.actors import Sensor
from packages.carla1s.tf import Point, CoordConverter, Transform, Coordinate

from src.dataset_dumper import DatasetDumper


class WaymoDumper(DatasetDumper):
    
    @dataclass
    class TimestampBind(DatasetDumper.SensorBind):
        """绑定时间戳至文件路径"""
        data_path: str
    
    @dataclass
    class PoseBind(DatasetDumper.SensorBind):
        """绑定位姿至文件路径"""
        data_path: str
    
    @dataclass 
    class LidarBind(DatasetDumper.SensorBind):
        """绑定激光雷达至输出点云文件夹路径"""
        data_path: str
        
    @dataclass
    class ImageBind(DatasetDumper.SensorBind):
        """绑定图像至输出文件夹路径"""
        data_path: str
        camera_name: str
    
    @dataclass
    class CalibTrBind(DatasetDumper.SensorBind):
        """绑定标定数据至文件路径"""
        data_path: str
    
    def __init__(self, 
                 root_path: str, 
                 *,
                 max_workers: int = 3):
        super().__init__(root_path, max_workers)
        self._timestamp_offset: Optional[float] = None
        self._pose_offset: Optional[Transform] = None
        self._pose_offset_coordinate: Optional[Coordinate] = None
        self.cameras = ["camera_FRONT","camera_FRONT_RIGHT","camera_BACK_RIGHT","camera_BACK","camera_BACK_LEFT","camera_FRONT_LEFT"]
    
    @property
    def current_frame_name(self) -> str:
        """当前帧的名称, 格式为timestamp_微秒"""
        return f'{self._current_frame_count-1:06d}'

    @contextmanager 
    def create_sequence(self, name: str = None):
        super().create_sequence(name)
        self._setup_content_folder()
        self.logger.info("=> WAYMO SEQUENCE BEGINS ".ljust(80, '='))
        yield
        self.logger.info("=> WAYMO SEQUENCE ENDS ".ljust(80, '='))

    def create_frame(self) -> 'WaymoDumper':
        self._current_frame_count += 1
        self._promises = []
        
        # 处理第一帧的特殊情况
        if self._current_frame_count == 1:
            self._timestamp_offset = None
            self._pose_offset = None
            
        for bind in self.binds:
            if isinstance(bind, self.LidarBind):
                self._promises.append(self.thread_pool.submit(self._dump_lidar, bind))
            elif isinstance(bind, self.ImageBind):
                self._promises.append(self.thread_pool.submit(self._dump_image, bind))
            elif isinstance(bind, self.TimestampBind):
                self._promises.append(self.thread_pool.submit(self._dump_timestamp, bind))
            elif isinstance(bind, self.PoseBind):
                self._promises.append(self.thread_pool.submit(self._dump_pose, bind))
                
        # 处理第一帧的特殊情况
        if self._current_frame_count == 1:
            self._setup_calib_file()

        return self

    def bind_camera(self,
                    sensor: Sensor,
                    *,
                    data_folder: str,
                    camera_name: str) -> 'DatasetDumper':
        """绑定相机,需指定camera_name(FRONT/FRONT_LEFT/FRONT_RIGHT/SIDE_LEFT/SIDE_RIGHT)"""
        if camera_name not in self.cameras:
            raise ValueError(f"Invalid camera_name: {camera_name}")
        self.logger.info(f"image data folder: {data_folder}")
        self._binds.append(self.ImageBind(sensor, data_folder, camera_name))
        return self

    def bind_lidar(self,
                   sensor: Sensor,
                   *,
                   data_folder: str) -> 'DatasetDumper':
        """绑定激光雷达"""
        self.logger.info(f"lidar data folder: {data_folder}")
        self._binds.append(self.LidarBind(sensor, data_folder))
        return self

    def bind_timestamp(self,
                       sensor: Sensor,
                       *,
                       file_path: str) -> 'DatasetDumper':
        """绑定时间戳"""
        if os.path.splitext(file_path)[1] == '':
            raise ValueError(f"Path {file_path} is a folder, not a file.")
        self.binds.append(self.TimestampBind(sensor, file_path))
        return self
    
    def bind_pose(self,
                  sensor: Sensor,
                  *,
                  file_path: str) -> 'DatasetDumper':
        """绑定位姿信息"""
        if os.path.splitext(file_path)[1] == '':
            raise ValueError(f"Path {file_path} is a folder, not a file.")
        self.binds.append(self.PoseBind(sensor, file_path))
        return self
    
    def bind_calib(self, 
                   tr_sensor: Sensor, 
                   *, 
                   file_path: str) -> 'DatasetDumper':
        if os.path.splitext(file_path)[1] == '':
            raise ValueError(f"Path {file_path} is a folder, not a file.")
        self.binds.append(self.CalibTrBind(tr_sensor, file_path))


    def _setup_content_folder(self):
        """创建内容文件夹"""
        for bind in self.binds:
            self.logger.info(f"data_path: {os.path.splitext(bind.data_path)}")
            if os.path.splitext(bind.data_path)[1] == '':
                os.makedirs(os.path.join(self.current_sequence_path, bind.data_path))
                self.logger.info(f"Created folder at: {os.path.join(self.current_sequence_path, bind.data_path)}")
            else:
                with open(os.path.join(self.current_sequence_path, bind.data_path), 'w') as f:
                    f.write('')
                    self.logger.info(f"Created file at: {os.path.join(self.current_sequence_path, bind.data_path)}")

    def _setup_calib_file(self):
        """创建标定文件."""
        # 寻找 calib bind
        calib_bind = next((bind for bind in self.binds if isinstance(bind, self.CalibTrBind)), None)
        
        if calib_bind is None:
            raise ValueError("Calib bind not found")
        
        self._promises.append(self.thread_pool.submit(self._dump_calib, calib_bind))

    def _dump_image(self, bind: ImageBind):
        # 阻塞等待传感器更新
        bind.sensor.on_data_ready.wait()
        
        # Waymo格式: timestamp_camera-name.jpg
        file_name = f"{self.current_frame_name}.png"
        path = os.path.join(self.current_sequence_path, bind.data_path, file_name)
        cv2.imwrite(path, bind.sensor.data.content)
        
        self.logger.debug(f"[frame={bind.sensor.data.frame}] Dumped {bind.camera_name} image to {path}")

    def _dump_lidar(self, bind: LidarBind):
        # 阻塞等待传感器更新
        bind.sensor.on_data_ready.wait()
        
        file_name = f"{self.current_frame_name}.ply"
        path = os.path.join(self.current_sequence_path, bind.data_path, file_name)
        
        # 转换点云格式
        points = bind.sensor.data.content[:, :4]  # x,y,z,intensity
        points = points.astype(np.float32)
        
        # Waymo使用右手坐标系,需要转换
        points[:, 1] = -points[:, 1]  # y轴反向
        
        # 保存为二进制文件
        # 使用open3d保存点云为PLY格式
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # 只使用xyz坐标
        o3d.io.write_point_cloud(path, pcd)
        
        self.logger.debug(f"[frame={bind.sensor.data.frame}] Dumped pointcloud to {path}")
        
    def _dump_timestamp(self, bind: DatasetDumper.SensorBind):
        """导出时间戳, 以纳秒为单位, 使用字符串格式.

        Args:
            bind (DatasetDumper.SensorBind): 参考的传感器绑定
        """
        bind.sensor.on_data_ready.wait()
        if self._timestamp_offset is None:
            self._timestamp_offset = bind.sensor.data.timestamp
        timestamp_ns = int((bind.sensor.data.timestamp - self._timestamp_offset) * 1e9)
        with open(os.path.join(self.current_sequence_path, bind.data_path), 'a') as f:
            f.write(f'{timestamp_ns}\n')
            
        self.logger.debug(f"[frame={bind.sensor.data.frame}] Dumped timestamp to {os.path.join(self.current_sequence_path, bind.data_path)}, value: {timestamp_ns}")

    def _dump_pose(self, bind: PoseBind):
        """导出位姿数据，使用4x4变换矩阵表示"""
        bind.sensor.on_data_ready.wait()
        
        path = os.path.join(self.current_sequence_path, bind.data_path)
        
        target = bind.sensor
        
        car_on_car_waymoori_waymocoord = (Coordinate(target.parent.get_transform())
                                             .change_orientation(CoordConverter.LEFT_HANDED_TO_RIGHT_HANDED_ORIENTATION))
        lidar_on_car_waymoori_waymocoord = (Coordinate(target.data.transform)
                                             .change_orientation(CoordConverter.LEFT_HANDED_TO_RIGHT_HANDED_ORIENTATION)
                                             .apply_transform(Transform(matrix=car_on_car_waymoori_waymocoord.data.matrix)))
        Tr = lidar_on_car_waymoori_waymocoord.data.matrix
        
        self.logger.info(f"relative pose: [frame={Tr}] ")
        
        # 获取位姿数据
        pose = bind.sensor.parent.get_transform()
        
        # 如果 offset 未设置, 则设置为当前帧的位姿
        if self._pose_offset is None:
            self._pose_offset = copy.deepcopy(pose)
            self._pose_offset_coordinate = Coordinate(self._pose_offset).change_orientation(CoordConverter.LEFT_HANDED_TO_RIGHT_HANDED_ORIENTATION)

        # # 计算当前帧的位姿相对初始帧的位姿

        vehicle_on_vehicle0_carlaori_carlacoord = (Coordinate(pose)
                                            .change_orientation(CoordConverter.LEFT_HANDED_TO_RIGHT_HANDED_ORIENTATION)
                                            .apply_transform(Transform(matrix=self._pose_offset_coordinate.data.matrix)))
        
        # 将位姿矩阵转换为 3x4 的变换矩阵
        pose_matrix = vehicle_on_vehicle0_carlaori_carlacoord.data.matrix[:3, :]

        # 横向展开, 表示为 1x12 的行向量, 并处理为小数点后 6 位的科学计数法表示, 以空格分隔
        pose_matrix = pose_matrix.flatten()
        pose_matrix = [f"{value:.6e}" for value in pose_matrix]
        pose_matrix = ' '.join(pose_matrix)
        
        with open(path, 'a') as f:
            f.write(f"{pose_matrix}\n")
            
        self.logger.debug(f"[frame={bind.sensor.data.frame}] Dumped pose to {path}")

    def _dump_calib(self, bind_calib: CalibTrBind):
        # 阻塞等待传感器更新
        bind_calib.sensor.on_data_ready.wait()
        # 准备对象
        target = bind_calib.sensor
        self.logger.info(f"target sensor type: {type(target)}")
        depth_cams = sorted([bind.sensor for bind in self.binds if isinstance(bind, self.ImageBind) 
                           and isinstance(bind.sensor, DepthCamera)], key=lambda cam: cam.attributes['role_name'])
        
        # 将激光雷达坐标系转换为Waymo坐标系
        car_on_car_waymoori_waymocoord = (Coordinate(target.parent.get_transform())
                                             .change_orientation(CoordConverter.LEFT_HANDED_TO_RIGHT_HANDED_ORIENTATION))
        lidar_on_car_waymoori_waymocoord = (Coordinate(target.data.transform)
                                             .change_orientation(CoordConverter.LEFT_HANDED_TO_RIGHT_HANDED_ORIENTATION)
                                             .apply_transform(Transform(matrix=car_on_car_waymoori_waymocoord.data.matrix)))
        Tr = lidar_on_car_waymoori_waymocoord.data.matrix
        
        self.logger.info(f"relative pose [frame={Tr}] ")
        
        # 准备储存路径
        path = os.path.join(self.current_sequence_path, bind_calib.data_path)
        
        def compute_intrinsic_matrix(w, h, fov):
            focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
            K = np.identity(3)
            K[0, 0] = K[1, 1] = focal
            K[0, 2] = w / 2.0
            K[1, 2] = h / 2.0
            K[2, 2] = 1
            return K
        
        # 处理相机
        for idx, cam in enumerate(depth_cams):
            # 获取内参
            image_width = int(cam.attributes['image_size_x'])
            image_height = int(cam.attributes['image_size_y'])
            fov = float(cam.attributes['fov'])
            
            K = compute_intrinsic_matrix(image_width, image_height, fov)
            
            # 获取外参
            cam_on_car_waymoori_waymocoord = (Coordinate(cam.data.transform)
                                               .change_orientation(CoordConverter.CARLA_CAM_TO_KITTI_CAM_ORIENTATION)
                                               .apply_transform(Transform(matrix=car_on_car_waymoori_waymocoord.data.matrix)))

            R = cam_on_car_waymoori_waymocoord.data.matrix[:3, :3]
            t = cam_on_car_waymoori_waymocoord.data.matrix[:3, -1].reshape(3, 1)
            RT = np.hstack((R, t))
            
            # 保存到文件
            with open(path, 'a') as calibfile:
                calibfile.write(f"RT{idx}:")
                string = ' '.join(['{:.12e}'.format(value) for row in RT for value in row])
                calibfile.write(string + "\n")
                calibfile.write(f"K{idx}:")
                string = ' '.join(['{:.12e}'.format(value) for row in K for value in row])
                calibfile.write(string + "\n")
                self.logger.debug(f"[frame={cam.data.frame}] Dumped calib RT{idx} and K{idx} to {path}")

        with open(path, 'a') as calibfile:
            calibfile.write("Tr:")
            string = ' '.join(['{:.12e}'.format(value) for row in Tr[:3, :] for value in row])
            calibfile.write(string + "\n")
            self.logger.debug(f"[frame={bind_calib.sensor.data.frame}] Dumped calib Tr to {path}")
            
    def process_depth_image(self, img_path, idx, output_depth_path, output_normal_path):
        # 遍历当前图片
        intrinsic = self.converter.intrinsic[idx]
        
        image_rgb = imageio.imread(img_path)[:,:,:3]
        image_bgr = image_rgb[..., ::-1]
        depth = depth_to_array(image_bgr)
        normal = depth_to_normal(depth, intrinsic)
        
        # 创建输出目录
        os.makedirs(output_depth_path, exist_ok=True)
        os.makedirs(output_normal_path, exist_ok=True)
        
        # 保存深度图
        npz_path = os.path.join(output_depth_path, str(os.path.splitext(img_path.name)[0]) + '.npz')
        np.savez_compressed(npz_path, depth[:,:, np.newaxis].astype(np.float32))
        
        # 保存法线图
        png_path = os.path.join(output_normal_path, str(os.path.splitext(img_path.name)[0]) + '.png')
        imageio.imwrite(png_path, normal)
        
    def process_semantic_image(self, img_path, seg_root_path, output_seg_path):
        image_rgb = imageio.imread(img_path)[:,:,:3]
        # 创建输出目录
        os.makedirs(output_seg_path, exist_ok=True)
        # 保存语义分割图
        npz_path = os.path.join(output_seg_path, str(os.path.splitext(img_path.name)[0]) + '.npz')
        np.savez_compressed(npz_path, image_rgb[:,:,0].astype(np.uint8))

    def postprocess(self):
        """后处理"""
        self.converter = WaymoConverter(self.current_sequence_path)
        self.converter.convert()
        
        print("start postprocess...")
        # 遍历相机
        for idx, cam in enumerate(self.cameras):
            print(f"postprocess camera: {cam} ")
            depth_root_path = Path(self.current_sequence_path) / 'images_depth' / cam
            seg_root_path = Path(self.current_sequence_path) / 'images_semantic' / cam
            output_depth_path = Path(self.current_sequence_path) / 'depths' / cam
            output_normal_path = Path(self.current_sequence_path) / 'normals' / cam
            output_seg_path = Path(self.current_sequence_path) / 'masks' / cam
            
            for img_path in depth_root_path.rglob('*'):
                if img_path.is_file():
                    self.process_depth_image(img_path, idx, output_depth_path, output_normal_path)
            
            for img_path in seg_root_path.rglob('*'):
                if img_path.is_file():
                    self.process_semantic_image(img_path, seg_root_path, output_seg_path)
