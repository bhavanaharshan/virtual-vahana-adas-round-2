import carla
import math

class RoutePlanner:
    def __init__(self, vehicle, world):
        self.vehicle = vehicle
        self.map = world.get_map()
        self.target_waypoint = None
        self.lookahead_distance = 5.0

    def get_target_waypoint(self):
        vehicle_transform = self.vehicle.get_transform()
        vehicle_loc = vehicle_transform.location
        veh_yaw = vehicle_transform.rotation.yaw
        
        # Snap to the nearest driving lane
        current_wp = self.map.get_waypoint(
            vehicle_loc, 
            project_to_road=True, 
            lane_type=carla.LaneType.Driving
        )
        
        # Look ahead to the next waypoints
        next_wps = current_wp.next(self.lookahead_distance)
        
        if next_wps:
            # INTERSECTION FIX: If there are multiple branches (left/right/straight),
            # pick the one that most closely matches the direction the car is already facing.
            if len(next_wps) > 1:
                def yaw_difference(wp):
                    wp_yaw = wp.transform.rotation.yaw
                    return abs((veh_yaw - wp_yaw + 180) % 360 - 180)
                
                # Sort the choices so the "straightest" path is always at index [0]
                next_wps.sort(key=yaw_difference)
                
            self.target_waypoint = next_wps[0]
            
        return self.target_waypoint