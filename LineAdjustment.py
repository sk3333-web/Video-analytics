import cv2
import numpy as np
import threading
import queue

# Global variables
lines = []
paused = False
frame = None
display_queue = queue.Queue()
target_size = None
dragging = False
selected_point = None
drag_threshold = 10  # Pixels

def initialize_target_size(width, height):
    global target_size
    target_size = (width, height)
    # Initialize default ROI line at 1/3 from bottom
    if not lines:
        default_height = int(height * 2/3)
        default_roi = [(0, default_height), (width, default_height)]
        lines.append(default_roi)

def get_closest_point(x, y):
    if not lines:
        return None, float('inf')
    
    line = lines[0]
    distances = [((px - x)**2 + (py - y)**2)**0.5 for px, py in line]
    min_dist = min(distances)
    if min_dist < drag_threshold:
        return distances.index(min_dist), min_dist
    return None, min_dist

def click_event(event, x, y, flags, param):
    global lines, dragging, selected_point
    
    if event == cv2.EVENT_LBUTTONDOWN:
        point_idx, dist = get_closest_point(x, y)
        if point_idx is not None:
            dragging = True
            selected_point = point_idx
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging and selected_point is not None:
            # Update point position
            lines[0] = list(lines[0])  # Convert tuple to list
            lines[0][selected_point] = (x, y)
            lines[0] = tuple(lines[0])  # Convert back to tuple
            print_roi_format()  # Print updated coordinates
            update_display()
    
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        selected_point = None

def print_roi_format():
    if lines:
        line = lines[0]  # Get the single ROI line
        print("\rROI Line: self.roi_line = [{}, {}]".format(line[0], line[1]), end='', flush=True)

def update_display():
    global frame
    if frame is not None:
        display_frame = frame.copy()
        # Draw the ROI line
        if lines:
            line = lines[0]
            cv2.line(display_frame, line[0], line[1], (0, 255, 0), 2)
            # Draw larger circles for draggable points
            cv2.circle(display_frame, line[0], 5, (0, 0, 255), -1)  # Start point
            cv2.circle(display_frame, line[1], 5, (0, 0, 255), -1)  # End point
        
        # Display line coordinates on the frame
        if lines:
            line = lines[0]
            text = f"ROI Line: Start {line[0]}, End {line[1]}"
            cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        display_queue.put(display_frame)

def process_stream(stream_url):
    global frame, paused
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"Failed to open camera stream: {stream_url}")
        return
    
    # Get video dimensions and initialize target size
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    initialize_target_size(width, height)
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to receive frame. Retrying...")
                continue
            update_display()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()

def display_frames(display_queue):
    cv2.namedWindow('Video')
    cv2.setMouseCallback('Video', click_event)
    
    while True:
        try:
            frame = display_queue.get(timeout=1)
            cv2.imshow('Video', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                global paused
                paused = not paused
            elif key == ord('r'):  # Reset to middle line
                if target_size:
                    lines[0] = [(0, target_size[1] * 2//3), (target_size[0], target_size[1] * 2//3)]
                    print_roi_format()
                    update_display()
            elif key == ord('s'):
                save_coordinates()
        except queue.Empty:
            pass
    
    cv2.destroyAllWindows()

def save_coordinates():
    if lines:
        with open('line_coordinates.txt', 'w') as f:
            line = lines[0]
            f.write(f"self.roi_line = [{line[0]}, {line[1]}]\n")
        print("\nROI line coordinates saved to line_coordinates.txt")
    else:
        print("\nNo lines to save")

def main():
    rtsp_url = "Rtsp link"
    
    # Start the stream processing thread
    stream_thread = threading.Thread(target=process_stream, args=(rtsp_url,))
    stream_thread.start()
    
    # Start the display thread
    display_thread = threading.Thread(target=display_frames, args=(display_queue,))
    display_thread.start()
    
    # Wait for threads to finish
    stream_thread.join()
    display_thread.join()

if __name__ == "__main__":
    main()