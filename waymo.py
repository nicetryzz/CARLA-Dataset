import logging
import time
import argparse

from packages.carla1s import CarlaContext, ManualExecutor, PassiveExecutor
from packages.carla1s.actors import Vehicle, RgbCamera, DepthCamera, SimpleLidar, SemanticCamera, BaseColorCamera
from packages.carla1s.tf import Transform

from src.waymo import WaymoDumper


def main(*, 
         fps: int = 20, 
         map: str = 'Town01', 
         output: str = './temp/', 
         host: str = 'localhost', 
         port: int = 2000,
         log_level: int = logging.DEBUG):

    with CarlaContext(host=host, port=port, log_level=log_level) as cc, ManualExecutor(cc, fixed_delta_seconds=1/fps) as exe:
        cc.reload_world(map_name=map)
        
        ego_vehicle: Vehicle = (cc.actor_factory
            .create(Vehicle, from_blueprint='vehicle.tesla.model3')
            .with_name("ego_vehicle")
            .with_transform(cc.get_spawn_point(57))
            .build())
        
        lidar_tf = Transform(x=0.00, y=0.00, z=2.20)
        camera_tf_list = [
            Transform(x=0.30, y=0.00, z=1.70, pitch=0, yaw=0, roll=0),
            Transform(x=0.30, y=0.50, z=1.70, pitch=0, yaw=60, roll=0),
            Transform(x=-1.20, y=0.50, z=1.70, pitch=0, yaw=120, roll=0),
            Transform(x=-1.20, y=0.00, z=1.70, pitch=0, yaw=180, roll=0),
            Transform(x=-1.20, y=-0.50, z=1.70, pitch=0, yaw=-120, roll=0),
            Transform(x=0.30, y=-0.50, z=1.70, pitch=0, yaw=-60, roll=0)
        ]

        lidar: SimpleLidar = (cc.actor_factory
            .create(SimpleLidar)
            .with_name("lidar")
            .with_transform(lidar_tf)
            .with_parent(ego_vehicle)
            .with_attributes(rotation_frequency=fps,
                             points_per_second=1000000,
                             channels=64,
                             range=100,
                             upper_fov=2,
                             lower_fov=-24.8,
                             )
            .build())
        
        exposure_list = [0,1]
        cameras = []
        for i, tf in enumerate(camera_tf_list):
            if len(exposure_list) > 0:
                cam_rgb = []
                for j, exposure in enumerate(exposure_list):
                    cam_rgb.append((cc.actor_factory
                        .create(RgbCamera)
                        .with_name(f"cam{i}_rgb_{j}")
                        .with_transform(tf)
                        .with_parent(ego_vehicle)
                        .with_attributes(image_size_x=1920, image_size_y=1080, fov=72,
                                         exposure_mode="manual", exposure_compensation=exposure,
                                         blur_amount=0, blur_radius=0.0, motion_blur_intensity=0.0,
                                         motion_blur_max_distortion=0.0, motion_blur_min_object_screen_size=0.0)
                        .build()))
            cam_depth: DepthCamera = (cc.actor_factory
                .create(DepthCamera)
                .with_name(f"cam{i}_depth")
                .with_transform(tf)
                .with_parent(ego_vehicle)
                .with_attributes(image_size_x=1920, image_size_y=1080, fov=72)
                .build())
            cam_semantic: SemanticCamera = (cc.actor_factory
                .create(SemanticCamera)
                .with_name(f"cam{i}_semantic")
                .with_transform(tf)
                .with_parent(ego_vehicle)
                .with_attributes(image_size_x=1920, image_size_y=1080, fov=72)
                .build())
            cam_basecolor: BaseColorCamera = (cc.actor_factory
                .create(BaseColorCamera)
                .with_name(f"cam{i}_basecolor")
                .with_transform(tf)
                .with_parent(ego_vehicle)
                .with_attributes(image_size_x=1920, image_size_y=1080, fov=72)
                .build())
            
            cameras.append({"cam_rgb": cam_rgb, "cam_depth": cam_depth, "cam_semantic": cam_semantic, "cam_basecolor": cam_basecolor})
            
        cc.all_actors_spawn().all_sensors_listen()
        exe.wait_ticks(1)
        
        ego_vehicle.set_autopilot(True)
        exe.wait_ticks(1)
        exe.wait_sim_seconds(1)
        
        dumper = WaymoDumper(output)

        dumper.bind_lidar(lidar, data_folder='lidar')
        
        cameras_names = ["camera_FRONT","camera_FRONT_RIGHT","camera_BACK_RIGHT","camera_BACK","camera_BACK_LEFT","camera_FRONT_LEFT"]
        for i, camera_pair in enumerate(cameras):
            
            camera_name = cameras_names[i]
            for j, cam_rgb in enumerate(camera_pair["cam_rgb"]):
                dumper.bind_camera(cam_rgb, data_folder=f'images/{cameras_names[i]}/expose_{j}', camera_name=cameras_names[i])
            dumper.bind_camera(camera_pair["cam_depth"], data_folder=f'images_depth/{cameras_names[i]}', camera_name=cameras_names[i])
            dumper.bind_camera(camera_pair["cam_semantic"], data_folder=f'images_semantic/{cameras_names[i]}', camera_name=cameras_names[i])
            dumper.bind_camera(camera_pair["cam_basecolor"], data_folder=f'images_basecolor/{cameras_names[i]}', camera_name=cameras_names[i])
        
        dumper.bind_calib(lidar, file_path="calib.txt")
        dumper.bind_timestamp(lidar, file_path="times.txt")
        dumper.bind_pose(lidar, file_path="poses.txt")
        
        # EXEC DUMP
        with dumper.create_sequence():
            for i in range(200):
                dumper.logger.info(f'-> FRAME: {dumper.current_frame_name} '.ljust(80, '-'))
                exe.wait_ticks(1)
                dumper.create_frame().join()
                
        dumper.postprocess()

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', type=int, default=20, help='Recommended FPS of the simulation')
    parser.add_argument('--map', type=str, default='SUSTech_COE_ParkingLot', help='Name of the map to load')
    parser.add_argument('--output', type=str, default='/home/hqlab/workspace/dataset/carla_data/dumper', help='Path to save the dataset')
    parser.add_argument('--host', type=str, default='localhost', help='Host of the Carla server')
    parser.add_argument('--port', type=int, default=2000, help='Port of the Carla server')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode, setting log level to DEBUG')
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.debug else logging.INFO
    
    try:
        main(fps=args.fps, map=args.map, output=args.output, host=args.host, port=args.port, log_level=log_level)
    except Exception:
        print(f'Exception occurred, check the log for more details.')

