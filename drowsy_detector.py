import cv2
import time
import pygame
import numpy as np
from ultralytics import YOLO
from datetime import datetime

# Initialize pygame for audio alerts
pygame.mixer.init()
try:
    alert_sound = pygame.mixer.Sound(r"C:\Users\Lenovo\Downloads\Driver Drowsiness Detection\Alert.wav")
except:
    print("Alert sound file not found. Using default system beep.")
    # Create a simple beep function as fallback
    def play_beep():
        pygame.mixer.Sound(pygame.sndarray.make_sound(np.sin(2 * np.pi * np.arange(10000) * 440 / 44100).astype(np.float32))).play()
    alert_sound = type('', (), {'play': play_beep})()

# Load the trained model
model = YOLO(r'C:\Users\Lenovo\Downloads\Driver Drowsiness Detection\model.pt')
  # change path if necessary

# Define class names based on your training data
class_names = ['Closed', 'Open', 'no_yawn', 'yawn']

# Initialize drowsiness detection parameters
eye_closed_counter = 0
yawn_counter = 0
drowsy_state = False
drowsy_duration = 0
drowsy_start_time = None
last_alert_time = 0
drowsy_alert_timeout = 3.0  # Time in seconds between alerts
drowsy_events = []  # To store drowsy detection events
alert_level = 0  # 0: Normal, 1: Warning, 2: Critical

# UI Parameters
frame_count = 0
fps = 0
fps_start_time = time.time()
start_time = time.time()
session_duration = 0

# Quit button tracking
quit_button_pressed = False
quit_button_hover = False
mouse_position = (0, 0)

# Modern color scheme (BGR format for OpenCV)
BG_COLOR = (40, 42, 54)       # Dark background
TEXT_COLOR = (248, 248, 242)  # Light text
PANEL_COLOR = (33, 34, 44)    # Darker panel
TITLE_BG = (68, 71, 90)       # Title background
HEADING_BG = (24, 26, 34)     # Heading background
BORDER_COLOR = (140, 140, 140)  # Border color
ALERT_COLORS = [
    (80, 250, 123),   # Green - Normal
    (241, 250, 140),  # Yellow - Warning
    (255, 85, 85)     # Red - Critical
]
BUTTON_COLOR = (95, 95, 175)     # Button color
BUTTON_HOVER_COLOR = (120, 120, 200)  # Button hover color

# Start video capture from webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get display resolution
ret, test_frame = cap.read()
if ret:
    frame_height, frame_width = test_frame.shape[:2]
else:
    frame_width, frame_height = 640, 480

# Create a single combined window with better proportions
# Add extra width to accommodate status panels on the right
window_width = frame_width + 300  # Additional space for side panel
window_height = frame_height + 60  # Just enough for title and margin

def ensure_uint8(img):
    """Ensure the image is in uint8 format for display"""
    if img.dtype != np.uint8:
        # Convert to uint8 if not already
        if np.max(img) <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def create_rounded_rectangle(img, top_left, bottom_right, radius, color, thickness=-1):
    """Draw a rounded rectangle"""
    img = ensure_uint8(img)
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    # Draw the filled rectangle
    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # Draw the four corners
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)
    else:
        # For non-filled rectangles
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    
    return img

def create_button(img, text, position, size, color, hover=False, text_color=(255, 255, 255)):
    """Create a professional looking button"""
    x, y = position
    width, height = size
    
    # Button background
    button_color = BUTTON_HOVER_COLOR if hover else BUTTON_COLOR
    
    # Create rounded button
    img = create_rounded_rectangle(img, (x, y), (x + width, y + height), 5, button_color, -1)
    
    # Add button border
    img = create_rounded_rectangle(img, (x, y), (x + width, y + height), 5, (200, 200, 200), 1)
    
    # Add text
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = x + (width - text_size[0]) // 2
    text_y = y + (height + text_size[1]) // 2
    
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    return img

def is_point_in_rect(point, rect_pos, rect_size):
    """Check if a point is inside a rectangle"""
    px, py = point
    rx, ry = rect_pos
    rw, rh = rect_size
    
    return rx <= px <= rx + rw and ry <= py <= ry + rh

def create_ui():
    """Create a unified UI canvas"""
    # Create a dark background for the entire UI
    ui = np.ones((window_height, window_width, 3), dtype=np.uint8) * np.array(BG_COLOR, dtype=np.uint8)
    
    # Create title banner at the top
    cv2.rectangle(ui, (0, 0), (window_width, 40), HEADING_BG, -1)
    
    # Add title text
    title_text = "DRIVER DROWSINESS DETECTION SYSTEM"
    text_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    title_x = (window_width - text_size[0]) // 2
    cv2.putText(ui, title_text, (title_x, 27), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)
    
    # Create area for the video feed
    video_area_x = 20
    video_area_y = 50  # Below the title
    
    # Draw a border around the video area
    cv2.rectangle(ui, 
                 (video_area_x - 2, video_area_y - 2), 
                 (video_area_x + frame_width + 2, video_area_y + frame_height + 2), 
                 BORDER_COLOR, 
                 2)
    
    return ui, (video_area_x, video_area_y)

def create_status_panel(ui, alert_level, eye_status, mouth_status, session_time):
    """Create a status panel on the right side"""
    global quit_button_hover
    
    # Status panel location
    panel_x = frame_width + 30
    panel_y = 50  # Same y as video
    panel_width = window_width - panel_x - 10
    panel_height = frame_height
    
    # Create panel background
    ui = create_rounded_rectangle(ui, 
                              (panel_x, panel_y), 
                              (panel_x + panel_width, panel_y + panel_height), 
                              10, PANEL_COLOR, -1)
    
    # Section title
    section_bg_height = 30
    cv2.rectangle(ui, 
                 (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + section_bg_height), 
                 TITLE_BG, -1)
    
    cv2.putText(ui, "DRIVER STATUS", 
                (panel_x + (panel_width - cv2.getTextSize("DRIVER STATUS", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]) // 2, 
                 panel_y + 22), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
    
    # Status indicators
    content_x = panel_x + 15
    content_y = panel_y + section_bg_height + 25
    
    # Eye status
    eye_color = ALERT_COLORS[0] if eye_status == "OPEN" else ALERT_COLORS[2]
    cv2.putText(ui, "Eye Status:", (content_x, content_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1)
    
    # Status indicator box
    status_box_width = 80
    status_box_x = panel_x + panel_width - 15 - status_box_width
    
    cv2.rectangle(ui, 
                 (status_box_x, content_y - 15), 
                 (status_box_x + status_box_width, content_y + 5), 
                 eye_color, -1)
    cv2.rectangle(ui, 
                 (status_box_x, content_y - 15), 
                 (status_box_x + status_box_width, content_y + 5), 
                 (255, 255, 255), 1)
    
    cv2.putText(ui, eye_status, 
                (status_box_x + (status_box_width - cv2.getTextSize(eye_status, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]) // 2, 
                 content_y - 3), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Mouth status
    content_y += 40
    mouth_color = ALERT_COLORS[0] if mouth_status == "NORMAL" else ALERT_COLORS[1]
    cv2.putText(ui, "Mouth Status:", (content_x, content_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1)
    
    cv2.rectangle(ui, 
                 (status_box_x, content_y - 15), 
                 (status_box_x + status_box_width, content_y + 5), 
                 mouth_color, -1)
    cv2.rectangle(ui, 
                 (status_box_x, content_y - 15), 
                 (status_box_x + status_box_width, content_y + 5), 
                 (255, 255, 255), 1)
    
    cv2.putText(ui, mouth_status, 
                (status_box_x + (status_box_width - cv2.getTextSize(mouth_status, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]) // 2, 
                 content_y - 3), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Quit button (replacing the drowsiness level bar)
    content_y += 40
    quit_button_width = 120
    quit_button_height = 30
    quit_button_x = panel_x + (panel_width - quit_button_width) // 2
    quit_button_y = content_y - 5
    
    # Check if mouse is hovering over the quit button
    button_hover = is_point_in_rect(mouse_position, 
                                   (quit_button_x, quit_button_y), 
                                   (quit_button_width, quit_button_height))
    quit_button_hover = button_hover
    
    # Create the quit button
    ui = create_button(ui, "QUIT", 
                      (quit_button_x, quit_button_y), 
                      (quit_button_width, quit_button_height), 
                      BUTTON_COLOR, 
                      hover=button_hover)
    
    # Separator line
    content_y += 50
    cv2.line(ui, 
             (panel_x + 10, content_y), 
             (panel_x + panel_width - 10, content_y), 
             (80, 80, 80), 1)
    
    # Section title: Metrics
    content_y += 30
    cv2.putText(ui, "SESSION METRICS", 
                (panel_x + (panel_width - cv2.getTextSize("SESSION METRICS", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]) // 2, 
                 content_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
    
    # Session time
    content_y += 35
    hours, remainder = divmod(int(session_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    cv2.putText(ui, "Session Duration:", (content_x, content_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1)
    cv2.putText(ui, time_str, (status_box_x, content_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1)
    
    # Drowsy events
    content_y += 30
    cv2.putText(ui, "Drowsy Events:", (content_x, content_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1)
    cv2.putText(ui, str(len(drowsy_events)), (status_box_x, content_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, ALERT_COLORS[min(len(drowsy_events), 2)], 1)
    
    # FPS
    content_y += 30
    cv2.putText(ui, "FPS:", (content_x, content_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1)
    cv2.putText(ui, f"{fps:.1f}", (status_box_x, content_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1)
    
    # Alert status
    content_y += 40
    alert_text = "NORMAL" if alert_level == 0 else "WARNING" if alert_level == 1 else "CRITICAL"
    
    # Status box background
    status_y = content_y - 15
    cv2.rectangle(ui, 
                 (panel_x + 10, status_y), 
                 (panel_x + panel_width - 10, status_y + 30), 
                 ALERT_COLORS[alert_level], -1)
    
    # Status text
    cv2.putText(ui, f"ALERT STATUS: {alert_text}", 
                (panel_x + (panel_width - cv2.getTextSize(f"ALERT STATUS: {alert_text}", 
                                                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]) // 2, 
                 content_y + 7), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Add current time at the bottom of panel
    current_time = datetime.now().strftime("%H:%M:%S")
    cv2.putText(ui, f"Time: {current_time}", 
                (panel_x + 10, panel_y + panel_height - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    
    return ui

# Mouse callback function for button interaction
def mouse_callback(event, x, y, flags, param):
    global mouse_position, quit_button_pressed
    
    mouse_position = (x, y)
    
    # Check for mouse click events
    if event == cv2.EVENT_LBUTTONDOWN:
        # Calculate quit button position (must match the position used in create_status_panel)
        panel_x = frame_width + 30
        panel_y = 50
        panel_width = window_width - panel_x - 10
        
        quit_button_width = 120
        quit_button_height = 30
        quit_button_x = panel_x + (panel_width - quit_button_width) // 2
        quit_button_y = panel_y + 30 + 25 + 40 + 40 - 5  # Match position in create_status_panel
        
        # Check if click is inside quit button
        if is_point_in_rect((x, y), 
                           (quit_button_x, quit_button_y), 
                           (quit_button_width, quit_button_height)):
            quit_button_pressed = True

try:
    window_name = "Driver Drowsiness Detection System"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Ensure frame is uint8
        frame = ensure_uint8(frame)
        
        # Create the UI base
        ui, (video_x, video_y) = create_ui()
        
        # Calculate FPS
        frame_count += 1
        if (time.time() - fps_start_time) > 1:
            fps = frame_count / (time.time() - fps_start_time)
            frame_count = 0
            fps_start_time = time.time()
        
        # Update session duration
        session_duration = time.time() - start_time
        
        # Create a copy of the frame for modifications
        display_frame = frame.copy()
        
        # Reset detection counters for current frame
        current_eye_closed = False
        current_yawning = False
        
        # Run inference
        try:
            results = model(frame, stream=True)
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = class_names[cls]
                    label = f"{class_name} {conf:.2f}"
                    
                    # Extract coordinates and ensure they are integers
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Ensure coordinates are within frame boundaries
                    x1 = max(0, min(xyxy[0], frame_width - 1))
                    y1 = max(0, min(xyxy[1], frame_height - 1))
                    x2 = max(0, min(xyxy[2], frame_width - 1))
                    y2 = max(0, min(xyxy[3], frame_height - 1))
                    
                    # Determine color based on class
                    if class_name == 'Closed':
                        color = (0, 0, 255)  # Red for closed eyes
                        current_eye_closed = True
                    elif class_name == 'yawn':
                        color = (0, 165, 255)  # Orange for yawning
                        current_yawning = True
                    else:
                        color = (0, 255, 0)  # Green for open eyes or no yawn
                    
                    # Draw detection box with nicer style
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label background with better styling
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    label_y1 = max(0, y1 - 20)
                    label_y2 = max(0, y1)
                    
                    # Semi-transparent background for label
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, 
                                 (x1, label_y1), 
                                 (x1 + label_size[0] + 10, label_y2), 
                                 color, -1)
                    cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
                    
                    # Draw label text
                    cv2.putText(display_frame, label, (x1 + 5, max(15, y1 - 5)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
        except Exception as e:
            print(f"Inference error: {e}")
        
        # Update drowsiness counters
        if current_eye_closed:
            eye_closed_counter = min(20, eye_closed_counter + 1)  # Cap at 20
        else:
            eye_closed_counter = max(0, eye_closed_counter - 1)
            
        if current_yawning:
            yawn_counter = min(20, yawn_counter + 1)  # Cap at 20
        else:
            yawn_counter = max(0, yawn_counter - 1)
        
        # Determine alert level
        previous_drowsy_state = drowsy_state
        
        if eye_closed_counter > 15 or yawn_counter > 15:
            alert_level = 2  # Critical
            drowsy_state = True
        elif eye_closed_counter > 8 or yawn_counter > 8:
            alert_level = 1  # Warning
            drowsy_state = True
        else:
            alert_level = 0  # Normal
            drowsy_state = False
        
        # If drowsy state just started, record the event
        if drowsy_state and not previous_drowsy_state:
            drowsy_events.append(datetime.now().strftime("%H:%M:%S"))
            drowsy_start_time = time.time()
        
        # If drowsy state just ended, record duration
        if not drowsy_state and previous_drowsy_state and drowsy_start_time is not None:
            drowsy_duration = time.time() - drowsy_start_time
            drowsy_start_time = None
        
        # Create status indicators for UI
        eye_status = "CLOSED" if eye_closed_counter > 8 else "OPEN"
        mouth_status = "YAWNING" if yawn_counter > 8 else "NORMAL"
        
        # Apply visual alerts to display frame
        if alert_level > 0:
            # Add a colored overlay based on alert level
            overlay = display_frame.copy()
            overlay_color = ALERT_COLORS[alert_level]
            cv2.rectangle(overlay, (0, 0), (frame_width, frame_height), 
                         (overlay_color[0], overlay_color[1], overlay_color[2]), -1)
            
            # Set overlay opacity based on alert level
            opacity = 0.1 if alert_level == 1 else 0.2
            cv2.addWeighted(overlay, opacity, display_frame, 1 - opacity, 0, display_frame)
            
            # Add alert text
            alert_text = "WARNING: DROWSINESS DETECTED" if alert_level == 1 else "CRITICAL: WAKE UP!"
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (frame_width - text_size[0]) // 2
            
            # Draw text with background
            overlay = display_frame.copy()
            cv2.rectangle(overlay, 
                         (text_x - 10, frame_height // 2 - 30), 
                         (text_x + text_size[0] + 10, frame_height // 2 + 10), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
            
            cv2.putText(display_frame, alert_text, (text_x, frame_height // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                       (overlay_color[0], overlay_color[1], overlay_color[2]), 2)
            
            # Sound alert with timeout
            current_time = time.time()
            if alert_level == 2 and current_time - last_alert_time > drowsy_alert_timeout:
                try:
                    alert_sound.play()
                    last_alert_time = current_time
                except Exception as e:
                    print(f"Sound alert error: {e}")
        
        # Ensure display_frame is uint8
        display_frame = ensure_uint8(display_frame)
        
        # Place the video frame in the UI
        try:
            if display_frame.shape[:2] == (frame_height, frame_width):
                ui[video_y:video_y + frame_height, video_x:video_x + frame_width] = display_frame
            else:
                resized_frame = cv2.resize(display_frame, (frame_width, frame_height))
                ui[video_y:video_y + frame_height, video_x:video_x + frame_width] = resized_frame
        except Exception as e:
            print(f"Error placing video frame in UI: {e}")
        
        # Update status panel
        ui = create_status_panel(ui, alert_level, eye_status, mouth_status, session_duration)
        
        # Ensure final UI is uint8
        ui = ensure_uint8(ui)
        
        # Display the combined UI
        cv2.imshow(window_name, ui)
        
        # Check for quit button press
        if quit_button_pressed:
            print("Quit button pressed - exiting application")
            break
        
        # Also keep keyboard exit (press 'q')
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()