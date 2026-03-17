import numpy as np
import cv2

class SensorFusion:
    def __init__(self):
        self.image_w = 800
        self.image_h = 600
        self.fov = 90
        self.focal = self.image_w / (2.0 * np.tan(self.fov * np.pi / 360.0))
        self.cx = self.image_w / 2.0
        self.cy = self.image_h / 2.0

    def fuse_lidar_to_camera(self, image, point_cloud, detections=None, yaw_offset=0.0):
        if point_cloud is None or len(point_cloud) == 0:
            return image

        # 1. HEIGHT FILTERING
        valid_points = point_cloud[
            (point_cloud[:, 2] > -3.0) & 
            (point_cloud[:, 2] < 1.0)    
        ]

        if len(valid_points) == 0:
            return image

        x_raw = valid_points[:, 0]
        y_raw = valid_points[:, 1]
        z_lidar = valid_points[:, 2]

        # 2. ROTATION MATRIX
        theta = np.radians(-yaw_offset)
        x_local = x_raw * np.cos(theta) - y_raw * np.sin(theta)
        y_local = x_raw * np.sin(theta) + y_raw * np.cos(theta)

        # 3. FORWARD FILTERING
        front_mask = x_local > 0.1
        x_local = x_local[front_mask]
        y_local = y_local[front_mask]
        z_lidar = z_lidar[front_mask]
        
        x_raw_f = x_raw[front_mask]
        y_raw_f = y_raw[front_mask]

        if len(x_local) == 0:
            return image

        # 4. PROJECTION MATH (3D to 2D)
        u = (self.focal * y_local / x_local) + self.cx
        v = (self.focal * -z_lidar / x_local) + self.cy

        valid_pixels = np.where((u >= 0) & (u < self.image_w) & (v >= 0) & (v < self.image_h))[0]

        for i in valid_pixels:
            px, py = int(u[i]), int(v[i])
            cv2.circle(image, (px, py), 4, (255, 0, 255), -1)

        # 5. BOUNDING BOX DEPTH ASSOCIATION & CLEAN COORDINATES
        if detections is not None:
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                
                inside_box_idx = valid_pixels[
                    (u[valid_pixels] >= x1) & (u[valid_pixels] <= x2) & 
                    (v[valid_pixels] >= y1) & (v[valid_pixels] <= y2)
                ]
                
                if len(inside_box_idx) > 0:
                    box_x = x_raw_f[inside_box_idx]
                    box_y = y_raw_f[inside_box_idx]
                    box_z = z_lidar[inside_box_idx]
                    
                    dist_array = np.sqrt(box_x**2 + box_y**2 + box_z**2)
                    closest_idx = np.argmin(dist_array)
                    
                    obj_x = box_x[closest_idx]
                    obj_y = box_y[closest_idx]
                    obj_z = box_z[closest_idx]
                    dist = dist_array[closest_idx]
                    
                    # SHIFT COORDINATES FOR INTUITIVE DISPLAY
                    ground_z = obj_z + 2.4  # Shift origin to road surface
                    lat_y = abs(obj_y)      # Absolute lateral distance
                    
                    text = f"{dist:.1f}m | X:{obj_x:.1f} Y:{lat_y:.1f} Z:{ground_z:.1f}"
                    cv2.putText(image, text, (x1, max(20, y1 - 25)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return image