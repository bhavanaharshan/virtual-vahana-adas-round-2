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
        
        # Get the closest lane waypoint
        current_wp = self.map.get_waypoint(vehicle_loc)
        
        # Sanity Check: Is this waypoint pointing completely backwards?
        # If the difference between car yaw and waypoint yaw is > 90 degrees,
        # it means we snapped to an oncoming lane. 
        wp_yaw = current_wp.transform.rotation.yaw
        yaw_diff = abs((veh_yaw - wp_yaw + 180) % 360 - 180)
        
        # If the lane is pointing the wrong way, we try to grab a waypoint slightly 
        # to the right to snap back to our correct driving lane.
        if yaw_diff > 90.0 and current_wp.get_right_lane() is not None:
            current_wp = current_wp.get_right_lane()

        # Project forward
        next_wps = current_wp.next(self.lookahead_distance)
        if next_wps:
            self.target_waypoint = next_wps[0]
            
        return self.target_waypoint