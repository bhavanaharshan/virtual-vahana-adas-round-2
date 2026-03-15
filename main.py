import cv2
import math
import carla
from utils.carla_utils import CarlaEnvironment
from core.perception import PerceptionModule
from core.control import VehicleController
from core.safety import SafetyModule
from core.planning import RoutePlanner

def get_speed(vehicle):
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

def main():
    env = CarlaEnvironment()
    perception = PerceptionModule()
    controller = VehicleController()
    safety = SafetyModule()
    # REMOVED planner from up here!
    
    try:
        ego_veh = env.spawn_ego_vehicle()
        if not ego_veh: return

        env.attach_camera()
        ego_veh.set_autopilot(False)
        
        # FIXED: Initialize Planner right here, passing BOTH the vehicle and the world
        planner = RoutePlanner(ego_veh, env.world)
        
        # --- NEW: Weather Presets ---
        weather_presets = [
            carla.WeatherParameters.ClearNoon,
            carla.WeatherParameters.CloudySunset,
            carla.WeatherParameters.HardRainNoon,
            carla.WeatherParameters.MidRainyNoon
        ]
        current_weather_idx = 0
        env.world.set_weather(weather_presets[current_weather_idx])
        print("Weather system initialized. Press 'w' to cycle weather.")
        
        cruise_speed = 20.0 
        
        # ... (The rest of your while True loop remains exactly the same) ...

        while True:
            if env.camera_data is not None:
                # 1. Perception
                raw_frame = env.camera_data
                annotated_frame, current_detections = perception.process_frame(raw_frame)
                
                # 2. Safety Arbitration
                aeb_active, final_frame = safety.evaluate_risk(current_detections, annotated_frame)
                
                # 3. Planning
                target_speed = 0.0 if aeb_active else cruise_speed
                target_wp = planner.get_target_waypoint()
                
                # 4. Control
                current_speed = get_speed(ego_veh)
                current_transform = ego_veh.get_transform()
                control_command = controller.run_step(target_speed, current_speed, current_transform, target_wp)
                ego_veh.apply_control(control_command)
                
                # Telemetry Update
                steer_val = control_command.steer
                status_color = (0, 0, 255) if aeb_active else (0, 255, 0)
                telemetry_speed = f"Speed: {int(current_speed)}/{int(target_speed)} km/h"
                telemetry_steer = f"Steer: {steer_val:.2f}"
                weather_text = f"Weather Mode: {current_weather_idx + 1}/4"
                
                cv2.putText(final_frame, telemetry_speed, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(final_frame, telemetry_steer, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(final_frame, weather_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                
                cv2.imshow("Virtual Vahana - Perception & Control", final_frame)
            
            # --- NEW: Keyboard Controls ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('w'):
                # Cycle to the next weather preset
                current_weather_idx = (current_weather_idx + 1) % len(weather_presets)
                env.world.set_weather(weather_presets[current_weather_idx])
                print(f"Weather changed to preset {current_weather_idx + 1}")
                
    finally:
        env.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()