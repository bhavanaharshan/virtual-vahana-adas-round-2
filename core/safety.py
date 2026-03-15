import cv2

class SafetyModule:
    def __init__(self):
        # Narrow the corridor back down to roughly one lane width
        self.path_x_min = 250
        self.path_x_max = 550
        
        # DEPTH ESTIMATION: If a bounding box is smaller than this, it's too far away to care about.
        self.critical_width = 120   # pixels
        self.critical_height = 100  # pixels
        
        # Ensure it's in the lower half of the screen
        self.critical_y_threshold = 350 

    def evaluate_risk(self, detections, frame):
        aeb_trigger = False
        warning_msg = ""
        
        # Draw the new "AEB Corridor" as two vertical lines for visual debugging
        cv2.line(frame, (self.path_x_min, 600), (self.path_x_min, self.critical_y_threshold), (0, 255, 255), 2)
        cv2.line(frame, (self.path_x_max, 600), (self.path_x_max, self.critical_y_threshold), (0, 255, 255), 2)
        cv2.putText(frame, "AEB CORRIDOR", (self.path_x_min + 10, self.critical_y_threshold - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            obj_class = det['class']
            
            box_center_x = (x1 + x2) // 2
            box_width = x2 - x1
            box_height = y2 - y1
            box_bottom_y = y2
            
            # Condition 1: Is the center of the object in our lane?
            in_path = self.path_x_min < box_center_x < self.path_x_max
            
            # Condition 2: Is it actually close to us? (Box must be large AND low on the screen)
            is_close = (box_width > self.critical_width or box_height > self.critical_height) and (box_bottom_y > self.critical_y_threshold)
            
            if in_path and is_close:
                aeb_trigger = True
                warning_msg = f"AEB ACTIVE: {obj_class.upper()} IMMINENT!"
                
                # Highlight the specific threat
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                break 
                
        if aeb_trigger:
            cv2.putText(frame, warning_msg, (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        
        return aeb_trigger, frame