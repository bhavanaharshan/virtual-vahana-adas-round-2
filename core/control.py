from simple_pid import PID
import carla
import math
import numpy as np

class VehicleController:
    def __init__(self, dt=0.05):
        # Longitudinal Controller (PID)
        self.lon_pid = PID(1.0, 0.1, 0.05, setpoint=0)
        self.lon_pid.output_limits = (-1.0, 1.0) 
        self.lon_pid.sample_time = dt
        
        # Lateral Controller (Stanley Gain)
        self.k_stanley = 0.25
        self.k_soft = 1.0

    def run_step(self, target_speed_kmh, current_speed_kmh, vehicle_transform, target_waypoint):
        control = carla.VehicleControl()
        
        # ==========================================
        # 1. LONGITUDINAL CONTROL (Throttle/Brake)
        # ==========================================
        if target_speed_kmh == 0.0:
            # EMERGENCY OVERRIDE: Bypass the PID controller entirely
            control.throttle = 0.0
            control.brake = 1.0  # Slam the brakes 100%
            self.lon_pid.reset() # Reset PID memory so it doesn't jerk forward when resuming
        else:
            # NORMAL DRIVING: Let the PID controller do its math
            self.lon_pid.setpoint = target_speed_kmh
            control_signal = self.lon_pid(current_speed_kmh)
            
            if control_signal > 0:
                control.throttle = min(control_signal, 1.0)
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = min(abs(control_signal), 1.0)

        # ==========================================
        # 2. LATERAL CONTROL (Stanley Steering)
        # ==========================================
        if target_waypoint is not None:
            # Vehicle kinematics
            veh_yaw = math.radians(vehicle_transform.rotation.yaw)
            wp_yaw = math.radians(target_waypoint.transform.rotation.yaw)
            v_meters_per_sec = max(current_speed_kmh / 3.6, 1.0) # Prevent division by zero
            
            # A. Calculate Heading Error (theta_e)
            heading_error = wp_yaw - veh_yaw
            # Normalize to [-pi, pi]
            heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

            # B. Calculate Cross-Track Error (e_fa)
            dx = target_waypoint.transform.location.x - vehicle_transform.location.x
            dy = target_waypoint.transform.location.y - vehicle_transform.location.y
            
            # FIXED: Flipped the terms to map correctly to CARLA's left-handed Y-axis
            crosstrack_error = dy * math.cos(veh_yaw) - dx * math.sin(veh_yaw)

            # C. Apply Stanley Control Law
            # C. Apply Stanley Control Law (with softening constant)
            # The + self.k_soft prevents the denominator from becoming too small
            steering_angle = heading_error + math.atan2(self.k_stanley * crosstrack_error, v_meters_per_sec + self.k_soft)
            # CARLA steering input is strictly bounded between -1.0 (left) and 1.0 (right)
            control.steer = float(np.clip(steering_angle, -1.0, 1.0))
        else:
            control.steer = 0.0

        return control