import carla
import random
import numpy as np
import cv2

class CarlaEnvironment:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.actor_list = []
        self.vehicle = None
        self.lidar_data = None
        # CHANGED: Now a dictionary to hold three camera feeds
        self.camera_data = {'center': None, 'left': None, 'right': None}

    def spawn_ego_vehicle(self, model='vehicle.tesla.model3'):
        blueprint = self.blueprint_library.find(model)
        blueprint.set_attribute('role_name', 'ego')
        
        # Get all valid spawn points and pick a random one
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        
        self.vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
        if self.vehicle is not None:
            self.actor_list.append(self.vehicle)
            print(f"Spawned ego vehicle: {self.vehicle.type_id}")
        else:
            print("Failed to spawn vehicle. Collision at spawn point.")
        
        return self.vehicle

    def attach_camera(self):
        if not self.vehicle: return

        cam_bp = self.blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '800')
        cam_bp.set_attribute('image_size_y', '600')
        cam_bp.set_attribute('fov', '90')
        
        # 1. Center Camera (Forward)
        center_trans = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.cam_center = self.world.spawn_actor(cam_bp, center_trans, attach_to=self.vehicle)
        self.cam_center.listen(lambda image: self._process_img(image, 'center'))

        # 2. Left Camera (Yaw = -90 degrees)
        left_trans = carla.Transform(carla.Location(x=1.0, y=-0.5, z=2.4), carla.Rotation(yaw=-90))
        self.cam_left = self.world.spawn_actor(cam_bp, left_trans, attach_to=self.vehicle)
        self.cam_left.listen(lambda image: self._process_img(image, 'left'))

        # 3. Right Camera (Yaw = 90 degrees)
        right_trans = carla.Transform(carla.Location(x=1.0, y=0.5, z=2.4), carla.Rotation(yaw=90))
        self.cam_right = self.world.spawn_actor(cam_bp, right_trans, attach_to=self.vehicle)
        self.cam_right.listen(lambda image: self._process_img(image, 'right'))
        
        self.actor_list.extend([self.cam_center, self.cam_left, self.cam_right])
        print("Surround Cameras attached (Center, Left, Right).")

    def _process_img(self, image, cam_pos):
        i = np.array(image.raw_data)
        i2 = i.reshape((600, 800, 4))
        i3 = i2[:, :, :3]
        self.camera_data[cam_pos] = i3
    
    def attach_lidar(self):
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('range', '50.0')
        lidar_bp.set_attribute('points_per_second', '100000')
        lidar_bp.set_attribute('rotation_frequency', '20')

        # Spawn at the exact same location as the center camera (z=2.4)
        lidar_trans = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_trans, attach_to=self.vehicle)
        self.lidar.listen(lambda data: self._process_lidar(data))
        self.actor_list.append(self.lidar)
        print("LiDAR attached for Sensor Fusion.")

    def _process_lidar(self, data):
        # Convert raw CARLA LiDAR data to a NumPy array of [X, Y, Z]
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        self.lidar_data = points[:, :3] # We only need X, Y, Z (dropping intensity)

    def cleanup(self):
        print("Cleaning up actors...")
        for actor in self.actor_list:
            if actor.is_alive:
                actor.destroy()
        print("Cleanup complete.")