import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import csv
from datetime import datetime
from collections import defaultdict, deque
import logging
import time
import threading
import queue
import argparse
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("camera_system.log")
    ]
)

class CameraSystem:
    def __init__(self, rtsp_urls):
        self.rtsp_urls = rtsp_urls
        self.cameras = {}
        self.camera_status = {}
        self.is_running = True
        self.system_lock = threading.Lock()
        self.camera_locks = {url: threading.Lock() for url in rtsp_urls}

        # Initialize status for all cameras
        for url in rtsp_urls:
            self.camera_status[url] = {
                'connected': False,
                'last_frame_time': None
            }

        # Initial connection attempt for all cameras
        self.initialize_cameras()

    def initialize_cameras(self):
        """Initialize all camera connections"""
        for url in self.rtsp_urls:
            try:
                with self.camera_locks[url]:
                    camera = cv2.VideoCapture(url)
                    if not camera.isOpened():
                        raise ConnectionError(f"Failed to connect to camera: {url}")
                    ret, frame = camera.read()
                    if not ret or frame is None:
                        raise ConnectionError(f"Failed to read initial frame from camera: {url}")

                    self.cameras[url] = camera
                    self.camera_status[url]['connected'] = True
                    self.camera_status[url]['last_frame_time'] = time.time()
                    logging.info(f"Successfully initialized camera: {url}")
            except Exception as e:
                logging.error(f"Failed to initialize camera {url}: {e}")
                self.cleanup_and_exit()

    def handle_disconnection(self, url):
        """Handle camera disconnection by immediately shutting down"""
        logging.error(f"FATAL: Camera {url} disconnected. Shutting down the system.")
        print(f"FATAL: Camera {url} disconnected. Shutting down the system.")

        # Force immediate shutdown
        with self.system_lock:
            self.is_running = False
            for cam_url, camera in self.cameras.items():
                try:
                    with self.camera_locks[cam_url]:
                        if camera is not None:
                            camera.release()
                            logging.info(f"Released camera: {cam_url}")
                except Exception as e:
                    logging.error(f"Error releasing camera {cam_url}: {e}")

            logging.error("FATAL: Force stopping all processes")
            os._exit(1)  # Immediately terminate all threads

    def get_frame(self, url):
        """Get a frame from specified camera with error handling"""
        if not self.is_running:
            return None, False

        try:
            with self.camera_locks[url]:
                camera = self.cameras.get(url)
                if camera is None:
                    logging.error(f"No camera object found for URL: {url}")
                    self.handle_disconnection(url)
                    return None, False

                if not camera.isOpened():
                    logging.error(f"Camera {url} is not opened.")
                    self.handle_disconnection(url)
                    return None, False

                ret, frame = camera.read()
                if not ret or frame is None or frame.size == 0:
                    logging.error(f"Failed to read frame from camera: {url}")
                    self.handle_disconnection(url)
                    return None, False

                # Successful frame read
                self.camera_status[url]['last_frame_time'] = time.time()
                return frame, True

        except Exception as e:
            logging.error(f"Error reading frame from camera {url}: {e}")
            self.handle_disconnection(url)
            return None, False

    def cleanup_and_exit(self):
        """Cleanup all resources and exit the program"""
        with self.system_lock:
            if not self.is_running:  # Prevent multiple cleanup attempts
                return

            self.is_running = False
            logging.error("FATAL: System shutdown initiated due to camera failure")
            print("FATAL: Camera system shutting down")  # Clear console message

            # Release all cameras
            for url, camera in self.cameras.items():
                try:
                    with self.camera_locks[url]:
                        if camera is not None:
                            camera.release()
                            logging.info(f"Released camera: {url}")
                except Exception as e:
                    logging.error(f"Error releasing camera {url}: {e}")

            logging.error("FATAL: All cameras released. Program exiting with error code 1")
            os._exit(1)  # Force exit the program

    def is_camera_active(self, url):
        """Check if a camera is active and usable"""
        return self.is_running and self.camera_status[url]['connected']

class ObjectTracking:
    def __init__(self, camera_system, stream_url, demographics_csv_path, csv_lock, display_queue,
                 show_video, enable_frame_skip, is_main_stream, image_save_dir, temp_grid_counts_path):
        self.camera_system = camera_system
        self.stream_url = stream_url
        self.camera_id = stream_url.split('/')[-1]
        self.demographics_csv_path = demographics_csv_path
        self.csv_lock = csv_lock
        self.display_queue = display_queue
        self.show_video = show_video
        self.enable_frame_skip = enable_frame_skip
        self.is_main_stream = is_main_stream
        self.image_save_dir = image_save_dir
        self.temp_grid_counts_path = temp_grid_counts_path
        logging.info(f"Camera {self.camera_id} using image save directory: {self.image_save_dir}")

        # Initialize YOLO models
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            weights_path = os.path.join(base_dir, 'yolov8n.pt')
            self.bytetrack_yaml_path = 'bytetrack.yaml'

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = YOLO(weights_path).to(self.device)
            self.model.fuse()

            if self.is_main_stream:
                self.face_model = YOLO('yolov8n-face.pt').to(self.device)
                self.gender_model = YOLO('best_yolov8_model_train24.pt').to(self.device)
                self.gender_threshold = 0.4
                logging.info("Face and gender models initialized for main stream")

        except Exception as e:
            logging.error(f"Error initializing models: {e}")
            self.camera_system.cleanup_and_exit()

        # Initialize tracking parameters
        self.target_size = (640, 480)
        self.grid_rows = 6
        self.grid_cols = 6
        self.grid_counts = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
        self.grid_ids = [[set() for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]

        # Initialize timers and counters
        self.start_time = time.time()
        self.interval = 60
        self.frame_skip = 3
        self.frame_count = 0

        # Initialize frame queue
        self.frame_queue = queue.Queue(maxsize=3)

        # Initialize main stream specific variables
        if self.is_main_stream:
            self.total_count_in = 0
            self.total_count_out = 0
            self.prev_positions = {}
            self.detected_persons = {}
            self.total_gender_age_count = {
                'Male': defaultdict(int),
                'Female': defaultdict(int)
            }
            self.roi_line = [(0, self.target_size[1] // 2),
                             (self.target_size[0], self.target_size[1] // 2)]
            self.line_crossers = set()
            self.window_size = 5        # Number of frames to consider in history
            self.confidence_threshold = 0.3  # Confidence threshold for considering predictions

        # Image saving parameters
        self.image_save_delay = 60
        self.image_saved = False

    def classify_age(self, age):
        if age <= 18:
            return "0_18"
        elif 19 <= age <= 24:
            return "19_24"
        elif 25 <= age <= 35:
            return "25_35"
        elif 36 <= age <= 55:
            return "36_55"
        else:
            return "55_plus"

    def detect_gender(self, person_crop, id):
        if not self.camera_system.is_running:
            return

        try:
            gender_results = self.gender_model(person_crop)
            if gender_results and len(gender_results[0].boxes) > 0:
                scores = gender_results[0].boxes.conf.cpu().numpy()
                classes = gender_results[0].boxes.cls.cpu().numpy().astype(int)
                confidence = max(scores)
                gender = 'Male' if classes[scores.argmax()] == 1 else 'Female'
            else:
                gender = 'Unknown'
                confidence = 0

            if id not in self.detected_persons:
                self.detected_persons[id] = {
                    'gender_history': deque(maxlen=self.window_size),
                    'male_confidence': 0.0,
                    'female_confidence': 0.0,
                    'recorded': False
                }

            person = self.detected_persons[id]
            person['gender_history'].append((gender, confidence))

            if confidence > self.confidence_threshold:
                if gender == 'Male':
                    person['male_confidence'] += confidence
                elif gender == 'Female':
                    person['female_confidence'] += confidence

            # Determine current gender based on accumulated confidences
            if person['male_confidence'] > person['female_confidence']:
                current_gender = 'Male'
            elif person['female_confidence'] > person['male_confidence']:
                current_gender = 'Female'
            else:
                # If confidences are equal, use the most frequent recent gender
                recent_genders = [g for g, c in person['gender_history'] if c > self.confidence_threshold]
                if recent_genders:
                    current_gender = max(set(recent_genders), key=recent_genders.count)
                else:
                    current_gender = 'Unknown'

            person['gender'] = current_gender
            logging.debug(f"Updated gender for person {id}: {current_gender}")

        except Exception as e:
            logging.error(f"Error in gender detection: {e}")

    def detect_age(self, person_crop, id):
        if not self.camera_system.is_running:
            return

        try:
            face_results = self.face_model(person_crop, device=self.device)
            if face_results and len(face_results[0].boxes) > 0:
                face_box = face_results[0].boxes[0].xyxy.cpu().numpy().astype(int)[0]
                face = person_crop[face_box[1]:face_box[3], face_box[0]:face_box[2]]
                analysis = DeepFace.analyze(face, actions=['age'], enforce_detection=False, silent=True)
                age = analysis[0]['age']
                age_class = self.classify_age(age)

                self.detected_persons[id]['age_class'] = age_class
                logging.info(f"Detected age for person {id}: {age}, Class: {age_class}")

        except Exception as e:
            logging.error(f"Error in age detection: {e}")

    def line_crossing(self, prev_point, current_point, line_start, line_end):
        return (prev_point[1] <= line_start[1] and current_point[1] > line_start[1]) or (
                prev_point[1] > line_start[1] and current_point[1] <= line_start[1])

    def save_raw_camera_image(self, frame):
        if not self.camera_system.is_running:
            return

        current_time = time.time()
        if not self.image_saved and current_time - self.start_time >= self.image_save_delay:
            try:
                image_sequence = "demo00001"
                filename = os.path.join(self.image_save_dir, f"{image_sequence}-{self.camera_id}.jpg")
                success = cv2.imwrite(filename, frame)
                if success:
                    logging.info(f"Saved raw image for Camera {self.camera_id}: {filename}")
                    self.image_saved = True
                else:
                    logging.error(f"Failed to save raw image for Camera {self.camera_id}: {filename}")
            except Exception as e:
                logging.error(f"Error saving raw image for Camera {self.camera_id}: {str(e)}")

    def update_latest_raw_image(self, frame):
        if not self.camera_system.is_running:
            return

        try:
            image_sequence = "demo00001"  # Reuse sequence number
            filename = os.path.join(self.image_save_dir, f"{image_sequence}-{self.camera_id}_latest_raw.jpg")
            success = cv2.imwrite(filename, frame)
            if success:
                logging.debug(f"Updated latest raw image for Camera {self.camera_id}: {filename}")
            else:
                logging.error(f"Failed to update latest raw image for Camera {self.camera_id}: {filename}")
        except Exception as e:
            logging.error(f"Error updating latest raw image for Camera {self.camera_id}: {str(e)}")

    def process_frame(self, frame):
        """Process a single frame with detection and tracking"""
        if not self.camera_system.is_running or frame is None or frame.size == 0:
            return None

        try:
            # Save raw frame
            self.save_raw_camera_image(frame)
            self.update_latest_raw_image(frame)

            # Prepare frame for processing
            frame = cv2.resize(frame, self.target_size)
            display_frame = frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.0

            with torch.no_grad():
                results = self.model.track(
                    source=frame_tensor,
                    persist=True,
                    tracker=self.bytetrack_yaml_path,
                    device=self.device,
                    classes=[0]  # Only detect persons
                )

                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    ids = results[0].boxes.id
                    if ids is not None:
                        ids = ids.cpu().numpy().astype(int)
                    else:
                        ids = []

                    for box, id in zip(boxes, ids):
                        # Draw bounding box and ID on display frame
                        cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), 2)
                        cv2.putText(display_frame, f"Id{id}", (box[0], box[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                        # Calculate grid position
                        center_point = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
                        grid_x = center_point[0] // (self.target_size[0] // self.grid_cols)
                        grid_y = center_point[1] // (self.target_size[1] // self.grid_rows)

                        # Update grid counts
                        if id not in self.grid_ids[grid_y][grid_x]:
                            self.grid_counts[grid_y, grid_x] += 1
                            self.grid_ids[grid_y][grid_x].add(id)

                        if self.is_main_stream:
                            # Gender detection
                            self.detect_gender(frame[box[1]:box[3], box[0]:box[2]], id)

                            # Line crossing detection
                            if id in self.prev_positions:
                                prev_point = self.prev_positions[id]
                                if self.line_crossing(prev_point, center_point, self.roi_line[0], self.roi_line[1]):
                                    if center_point[1] > prev_point[1]:
                                        self.total_count_in += 1
                                        if id not in self.line_crossers:
                                            self.detect_age(frame[box[1]:box[3], box[0]:box[2]], id)
                                            self.line_crossers.add(id)
                                    else:
                                        self.total_count_out += 1

                                    if id in self.detected_persons and not self.detected_persons[id]['recorded']:
                                        demographics = self.detected_persons[id]
                                        gender = demographics.get('gender', 'Unknown')
                                        age_class = demographics.get('age_class', 'Unknown')
                                        if gender != 'Unknown' and age_class != 'Unknown':
                                            self.total_gender_age_count[gender][age_class] += 1
                                            self.detected_persons[id]['recorded'] = True
                                            logging.info(f"Recorded demographics for person {id} crossing line: {demographics}")

                            self.prev_positions[id] = center_point

                            # Add gender and age information to display frame
                            if id in self.detected_persons:
                                demographics = self.detected_persons[id]
                                gender = demographics.get('gender', 'Unknown')
                                age_class = demographics.get('age_class', 'Unknown')
                                cv2.putText(display_frame, f"Id{id}: {gender}, {age_class}",
                                            (box[0], box[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                    if self.is_main_stream:
                        # Clean up detected_persons dictionary
                        tracked_ids = set(ids)
                        ids_to_remove = [pid for pid in self.detected_persons if pid not in tracked_ids]
                        for pid in ids_to_remove:
                            del self.detected_persons[pid]

            # Draw grid lines
            for i in range(1, self.grid_rows):
                y = i * (self.target_size[1] // self.grid_rows)
                cv2.line(display_frame, (0, y), (self.target_size[0], y), (0, 255, 0), 1)
            for i in range(1, self.grid_cols):
                x = i * (self.target_size[0] // self.grid_cols)
                cv2.line(display_frame, (x, 0), (x, self.target_size[1]), (0, 255, 0), 1)

            # Add grid counts
            for i in range(self.grid_rows):
                for j in range(self.grid_cols):
                    x = j * (self.target_size[0] // self.grid_cols) + 5
                    y = i * (self.target_size[1] // self.grid_rows) + 20
                    cv2.putText(display_frame, str(self.grid_counts[i, j]), (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw ROI line and counts for main stream
            if self.is_main_stream:
                cv2.line(display_frame, self.roi_line[0], self.roi_line[1], (0, 255, 0), 2)
                cv2.putText(display_frame, f"In: {self.total_count_in}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Out: {self.total_count_out}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Check if it's time to save counts
            if time.time() - self.start_time >= self.interval:
                self.save_counts()

            logging.debug(f"Processed frame for Camera {self.camera_id}")
            return display_frame  # Return the display frame with all visual elements
        except Exception as e:
            logging.error(f"Error processing frame for camera {self.camera_id}: {e}")
            self.camera_system.handle_disconnection(self.stream_url)
            return None

    def save_counts(self):
        """Save detection counts and demographics to CSV files"""
        if not self.camera_system.is_running:
            return

        try:
            current_time = int(time.time())

            # Save grid counts
            with self.csv_lock:
                csv_row = [current_time, self.camera_id] + list(self.grid_counts.flatten())
                with open(self.temp_grid_counts_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(csv_row)
                logging.info(f"Wrote grid counts for camera {self.camera_id} to {self.temp_grid_counts_path}")

                # Save demographics for main stream
                if self.is_main_stream:
                    male_counts = {k: self.total_gender_age_count['Male'][k]
                                   for k in self.total_gender_age_count['Male']}
                    female_counts = {k: self.total_gender_age_count['Female'][k]
                                     for k in self.total_gender_age_count['Female']}

                    demographics_row = [
                        current_time, self.camera_id,
                        self.total_count_in, self.total_count_out,
                        male_counts.get("0_18", 0), male_counts.get("19_24", 0), male_counts.get("25_35", 0),
                        male_counts.get("36_55", 0), male_counts.get("55_plus", 0),
                        female_counts.get("0_18", 0), female_counts.get("19_24", 0),
                        female_counts.get("25_35", 0), female_counts.get("36_55", 0),
                        female_counts.get("55_plus", 0)
                    ]

                    with open(self.demographics_csv_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(demographics_row)
                    logging.info(f"Wrote demographics to {self.demographics_csv_path}")

            logging.info(f"Data saved to CSVs for camera {self.camera_id}")

            # Reset counters and data structures
            self.reset_counters()
            logging.info(f"Reset counters for camera {self.camera_id}")
        except Exception as e:
            logging.error(f"Error saving counts for camera {self.camera_id}: {e}")
            self.camera_system.cleanup_and_exit()

    def reset_counters(self):
        """Reset all counters and tracking data"""
        self.grid_counts.fill(0)
        self.grid_ids = [[set() for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]

        if self.is_main_stream:
            self.total_count_in = 0
            self.total_count_out = 0
            self.prev_positions.clear()
            self.detected_persons.clear()
            self.total_gender_age_count = {'Male': defaultdict(int), 'Female': defaultdict(int)}
            self.line_crossers.clear()

        self.start_time = time.time()

    def process_stream(self):
        """Main processing loop for the camera stream"""
        while self.camera_system.is_running:
            frame, active = self.camera_system.get_frame(self.stream_url)

            if not active:
                continue

            self.frame_count += 1
            if self.enable_frame_skip and self.frame_count % self.frame_skip != 0:
                continue

            if frame is not None:
                processed_frame = self.process_frame(frame)
                if self.show_video and processed_frame is not None:
                    self.display_queue.put((self.camera_id, processed_frame))

def display_frames(display_queue):
    windows = {}

    while True:
        try:
            camera_id, frame = display_queue.get(timeout=1)
            window_name = f"Camera {camera_id}"

            if frame is None or frame.size == 0:
                logging.warning(f"Received empty frame for camera {camera_id}. Skipping display.")
                continue

            if window_name not in windows:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 640, 480)
                windows[window_name] = True

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Exit signal received. Shutting down display.")
                break
        except queue.Empty:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Exit signal received during queue empty. Shutting down display.")
                break

    for window in windows:
        cv2.destroyWindow(window)

class MultiStreamObjectTracking:
    def __init__(self, stream_urls, show_video, enable_frame_skip):
        self.stream_urls = stream_urls
        self.show_video = show_video
        self.enable_frame_skip = enable_frame_skip

        # Initialize the camera system
        self.camera_system = CameraSystem(stream_urls)

        # Initialize directories and paths
        self.base_dir = os.getcwd()
        self.grid_counts_csv_dir = os.path.join(self.base_dir, 'grid_counts_csv')
        self.demographics_csv_dir = os.path.join(self.base_dir, 'demographics_csv')
        self.image_save_dir = os.path.join(self.base_dir, 'camera_images')

        # Create necessary directories
        self.create_directories()

        # Initialize CSV paths and locks
        self.demographics_csv_path = os.path.join(self.base_dir, 'demographics.csv')
        self.temp_grid_counts_path = 'temp_grid_counts_all.csv'
        self.csv_lock = threading.Lock()

        # Initialize display queue and threads list
        self.display_queue = queue.Queue()
        self.threads = []

        # Initialize rotation parameters
        self.rotation_interval = 300  # 5 minutes
        self.last_rotation_time = time.time()

        # Initialize CSV files
        self.initialize_csv_files()

    def create_directories(self):
        """Create necessary directories for the system"""
        directories = [
            self.grid_counts_csv_dir,
            self.demographics_csv_dir,
            self.image_save_dir
        ]

        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                logging.info(f"Directory created/verified: {directory}")
            except Exception as e:
                logging.error(f"Error creating directory {directory}: {e}")
                self.camera_system.cleanup_and_exit()

    def initialize_csv_files(self):
        """Initialize CSV files with empty state"""
        try:
            # Initialize demographics CSV
            open(self.demographics_csv_path, 'w').close()
            logging.info(f"Initialized demographics CSV: {self.demographics_csv_path}")

            # Initialize temporary grid counts CSV
            open(self.temp_grid_counts_path, 'w').close()
            logging.info(f"Initialized grid counts CSV: {self.temp_grid_counts_path}")
        except Exception as e:
            logging.error(f"Error initializing CSV files: {e}")
            self.camera_system.cleanup_and_exit()

    def rotation_timer(self):
        """Timer thread to rotate CSV files at set intervals"""
        while True:
            if not self.camera_system.is_running:
                return
            try:
                time.sleep(10)
                current_time = time.time()
                if current_time - self.last_rotation_time >= self.rotation_interval:
                    logging.info("Rotation interval reached. Starting CSV rotation.")
                    self.rotate_csv_files()
                    self.last_rotation_time = current_time
                else:
                    time_remaining = self.rotation_interval - (current_time - self.last_rotation_time)
                    logging.debug(f"Next rotation in {time_remaining:.2f} seconds")
            except Exception as e:
                logging.error(f"Error in rotation timer: {e}")
                self.camera_system.cleanup_and_exit()

    def rotate_csv_files(self):
        """Rotate and archive CSV files"""
        if not self.camera_system.is_running:
            return

        try:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Rotate grid counts file
            if os.path.exists(self.temp_grid_counts_path):
                combined_file = f"grid_counts_combined_{current_time}.csv"
                combined_file_path = os.path.join(self.grid_counts_csv_dir, combined_file)
                os.rename(self.temp_grid_counts_path, combined_file_path)
                logging.info(f"Rotated grid counts to: {combined_file_path}")

                # Create new empty temporary file
                open(self.temp_grid_counts_path, 'w').close()

            # Rotate demographics file
            if os.path.exists(self.demographics_csv_path):
                demographics_file = f"demographics_{current_time}.csv"
                demographics_final_path = os.path.join(self.demographics_csv_dir, demographics_file)
                os.rename(self.demographics_csv_path, demographics_final_path)
                logging.info(f"Rotated demographics file to: {demographics_final_path}")

                # Create new empty demographics file
                open(self.demographics_csv_path, 'w').close()
                logging.info(f"Created new empty demographics file: {self.demographics_csv_path}")

        except Exception as e:
            logging.error(f"Error rotating CSV files: {e}")
            self.camera_system.cleanup_and_exit()

    def run(self):
        """Start the multi-stream tracking system"""
        try:
            # Start object tracking for each camera
            for i, stream_url in enumerate(self.stream_urls):
                is_main_stream = (i == 0)  # First camera is main stream

                tracker = ObjectTracking(
                    self.camera_system,
                    stream_url,
                    self.demographics_csv_path,
                    self.csv_lock,
                    self.display_queue,
                    self.show_video,
                    self.enable_frame_skip,
                    is_main_stream,
                    self.image_save_dir,
                    self.temp_grid_counts_path
                )

                thread = threading.Thread(target=tracker.process_stream, daemon=True)
                self.threads.append(thread)
                thread.start()
                logging.info(f"Started tracking thread for camera {stream_url}")

            # Start rotation timer thread
            rotation_thread = threading.Thread(target=self.rotation_timer, daemon=True)
            rotation_thread.start()
            self.threads.append(rotation_thread)
            logging.info("Started rotation timer thread")

            # Start display thread if video display is enabled
            if self.show_video:
                display_thread = threading.Thread(target=display_frames, args=(self.display_queue,), daemon=True)
                display_thread.start()
                self.threads.append(display_thread)
                logging.info("Started display thread")

            # Keep the main thread alive to allow daemon threads to run
            while self.camera_system.is_running:
                time.sleep(1)

        except Exception as e:
            logging.error(f"Error in MultiStreamObjectTracking: {e}")
            self.camera_system.cleanup_and_exit()

def run_multi_stream_object_tracking(show_video, enable_frame_skip):
    """Main entry point for the tracking system"""
    rtsp_urls = ["RTSP_LINKS"]

    try:
        multi_tracker = MultiStreamObjectTracking(rtsp_urls, show_video, enable_frame_skip)
        multi_tracker.run()
    except Exception as e:
        logging.error(f"Fatal error in main program: {e}")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run multi-stream object tracking')
    parser.add_argument('--show_video', action='store_true', help='Display video output')
    parser.add_argument('--enable_frame_skip', action='store_true', help='Enable frame skipping')
    args = parser.parse_args()

    run_multi_stream_object_tracking(args.show_video, args.enable_frame_skip)
