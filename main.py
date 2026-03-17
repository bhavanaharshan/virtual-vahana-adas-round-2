import cv2
import math
import carla
from utils.carla_utils import CarlaEnvironment
from core.perception import PerceptionModule
from core.control import VehicleController
from core.safety import SafetyModule
from core.planning import RoutePlanner
from core.fusion import SensorFusion

def get_speed(vehicle):
    """Helper function to calculate speed in km/h from CARLA velocity vector."""
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

def main():
    # 1. Initialize all modules
    env = CarlaEnvironment()
    perception = PerceptionModule()
    controller = VehicleController()
    safety = SafetyModule()
    fusion = SensorFusion()
    
    try:
        # 2. Spawn and setup ego vehicle
        ego_veh = env.spawn_ego_vehicle()
        if not ego_veh: 
            print("Failed to spawn ego vehicle.")
            return

        env.attach_camera()
        env.attach_lidar()
        ego_veh.set_autopilot(False)
        
        # Initialize Planner, passing BOTH the vehicle and the world
        planner = RoutePlanner(ego_veh, env.world)
        
        # 3. Weather Presets
        weather_presets = [
            carla.WeatherParameters.ClearNoon,
            carla.WeatherParameters.CloudySunset,
            carla.WeatherParameters.HardRainNoon,
            carla.WeatherParameters.MidRainyNoon
        ]
        current_weather_idx = 0
        env.world.set_weather(weather_presets[current_weather_idx])
        print("Weather system initialized. Press 'w' to cycle weather. Press 'q' to quit.")
        
        cruise_speed = 20.0 

        # 4. Main Simulation Loop
        while True:
            # Wait until all cameras have a frame
            if all(env.camera_data[pos] is not None for pos in ['center', 'left', 'right']):
                
                # A. PERCEPTION: Run YOLO on all three camera feeds first
                annotated_left, left_dets = perception.process_frame(env.camera_data['left'])
                annotated_center, center_dets = perception.process_frame(env.camera_data['center'])
                annotated_right, right_dets = perception.process_frame(env.camera_data['right'])
                
                # B. SENSOR FUSION: Overlay 3D LiDAR data and bounding box telemetry
                annotated_left = fusion.fuse_lidar_to_camera(annotated_left, env.lidar_data, left_dets, yaw_offset=-90)
                annotated_center = fusion.fuse_lidar_to_camera(annotated_center, env.lidar_data, center_dets, yaw_offset=0)
                annotated_right = fusion.fuse_lidar_to_camera(annotated_right, env.lidar_data, right_dets, yaw_offset=90)
                
                # C. SPEED CALCULATION (Required for Dynamic AEB)
                current_speed = get_speed(ego_veh)
                current_transform = ego_veh.get_transform()

                # D. SAFETY ARBITRATION: Pass the LiDAR, Center Camera, and Speed to the AEB
                aeb_active, final_center = safety.evaluate_risk(center_dets, annotated_center, env.lidar_data, current_speed)

                # E. PLANNING AND CONTROL (Strictly Autonomous)
                target_speed = 0.0 if aeb_active else cruise_speed
                target_wp = planner.get_target_waypoint()
                control_command = controller.run_step(target_speed, current_speed, current_transform, target_wp)
                
                # Lock the tires if AEB triggers to prevent skidding
                if aeb_active:
                    control_command.hand_brake = True
                
                ego_veh.apply_control(control_command)
                
                # ==========================================
                # F. 270-DEGREE PANORAMIC STITCH
                # ==========================================
                # Concatenate the images side-by-side seamlessly
                panorama = cv2.hconcat([annotated_left, final_center, annotated_right])
                
                # Resize to fit comfortably on your screen
                display_pano = cv2.resize(panorama, (1800, 450))
                
                # Draw Telemetry Update
                steer_val = control_command.steer
                status_color = (0, 0, 255) if aeb_active else (0, 255, 0)
                
                cv2.putText(display_pano, f"Speed: {int(current_speed)}/{int(target_speed)} km/h", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
                cv2.putText(display_pano, f"Steer: {steer_val:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                cv2.putText(display_pano, f"Weather Mode: {current_weather_idx + 1}/4", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
                
                cv2.imshow("Virtual Vahana - 270-Degree FOV", display_pano)
            
            # G. KEYBOARD CONTROLS (Only Weather and Quit remain)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('w'):
                current_weather_idx = (current_weather_idx + 1) % len(weather_presets)
                env.world.set_weather(weather_presets[current_weather_idx])
                
    finally:
        env.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()