import cv2
import numpy as np
import time

class SafetyModule:
    def __init__(self):
        self.bumper_offset = 2.5   
        self.lane_width_m = 2.0    # WIDENED: Up from 1.2m to catch vehicles crossing our nose!
        self.height_max = 1.0      
        self.height_min = -2.0     
        
        # MEMORY: Track distance and time to calculate True Closing Velocity
        self.prev_min_dist = float('inf')
        self.prev_time = time.time()

    def evaluate_risk(self, detections, frame, point_cloud, current_speed_kmh):
        aeb_trigger = False
        warning_msg = ""
        
        # Calculate time delta for physics math
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt <= 0: dt = 0.001 # Prevent division by zero
        
        if point_cloud is not None:
            # Look up to 20 meters ahead, with a wider lane check
            emergency_points = point_cloud[
                (point_cloud[:, 0] > self.bumper_offset) & (point_cloud[:, 0] < 20.0) &  
                (point_cloud[:, 1] > -self.lane_width_m) & (point_cloud[:, 1] < self.lane_width_m) & 
                (point_cloud[:, 2] > self.height_min) & (point_cloud[:, 2] < self.height_max)
            ]
            
            if len(emergency_points) > 15:
                # 1. Get current distance
                current_min_dist = np.min(emergency_points[:, 0]) - self.bumper_offset
                
                # 2. TRUE CLOSING VELOCITY
                # How many meters did the gap shrink by in the last fraction of a second?
                delta_dist = self.prev_min_dist - current_min_dist
                
                # If delta_dist is positive, the obstacle is getting closer to us
                closing_speed_ms = delta_dist / dt if self.prev_min_dist != float('inf') else 0.0
                
                # 3. Calculate True Time-To-Collision (TTC) based on closing speed, not ego speed!
                ttc = current_min_dist / closing_speed_ms if closing_speed_ms > 0.5 else float('inf')
                
                # 4. The Trigger Logic
                # If the gap is shrinking and we will hit it in less than 2.5 seconds...
                # OR if the object breaches a strict 4.0-meter panic bubble
                if (closing_speed_ms > 0.5 and ttc < 2.5) or current_min_dist < 4.0:
                    aeb_trigger = True
                    warning_msg = "UR MOVING TOWARDS OBSTACLE: STOPPING"
                
                # Save state for the next frame
                self.prev_min_dist = current_min_dist
            else:
                self.prev_min_dist = float('inf')

        self.prev_time = current_time

        # Draw the visual warning
        if aeb_trigger:
            cv2.putText(frame, warning_msg, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
                        
        return aeb_trigger, frame