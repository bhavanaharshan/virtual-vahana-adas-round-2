import carla
import random
import numpy as np
import cv2

class CarlaEnvironment:
    def __init__(self):
        # Connect to the CARLA server
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.actor_list = []
        self.vehicle = None
        self.camera_data = None

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
        if not self.vehicle:
            print("No vehicle to attach camera to!")
            return

        cam_bp = self.blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '800')
        cam_bp.set_attribute('image_size_y', '600')
        cam_bp.set_attribute('fov', '90')
        
        # Position the camera on the roof/windshield area
        spawn_point = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(cam_bp, spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        
        # Listen to sensor data and pass it to our processing function
        self.camera.listen(lambda image: self._process_img(image))
        print("RGB Camera attached.")

    def _process_img(self, image):
        # Convert raw CARLA sensor data to a numpy array for OpenCV
        i = np.array(image.raw_data)
        i2 = i.reshape((600, 800, 4))  # RGBA format
        i3 = i2[:, :, :3]              # Drop Alpha channel (RGB)
        self.camera_data = i3

    def cleanup(self):
        print("Cleaning up actors...")
        for actor in self.actor_list:
            if actor.is_alive:
                actor.destroy()
        print("Cleanup complete.")