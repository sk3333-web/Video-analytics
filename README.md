# Video-analytics
This advanced retail analytics solution leverages computer vision and deep learning to provide comprehensive store insights
Video-Analytics
Overview
This advanced retail analytics solution leverages computer vision and deep learning to provide comprehensive store insights. The system uses real-time video feeds to detect and track individuals, predict demographics, and perform grid-based analysis to understand consumer behavior.

Features
Real-Time Object Tracking: Tracks people across multiple video streams using YOLO and ByteTrack.
Demographics Analysis: Predicts age and gender using YOLO models and DeepFace integration.
Dynamic Line Adjustment: Allows users to customize Region of Interest (ROI) lines for specific analytics needs.
Grid-Based Insights: Provides grid-level counts for enhanced spatial analysis.
Multi-Camera Support: Processes multiple RTSP streams simultaneously.
Data Logging: Saves demographic and grid-based analytics to CSV files for further analysis.
System Requirements
Hardware: NVIDIA GPU with CUDA support
Software:
Python 3.8+
OpenCV
PyTorch with CUDA
YOLOv8 and ByteTrack
DeepFace
Installation
Clone this repository:

bash
Copy code
git clone https://github.com/<username>/Video-analytics.git
cd Video-analytics
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Download the YOLO model weights and place them in the root directory:

yolov8n.pt (Object detection)
yolov8n-face.pt (Face detection)
best_yolov8_model_train24.pt (Gender detection)
Ensure the line_coordinates.txt file exists for custom ROI line adjustments.

Usage
1. Running the Multi-Camera Tracking System
Run the multi-stream object tracking system:

bash
Copy code
python Video_Analytics.py --show_video --enable_frame_skip
Arguments:
--show_video: Display live video feed with annotations.
--enable_frame_skip: Skip frames to optimize processing speed.
2. Adjusting the ROI Line
To customize the Region of Interest (ROI) line:

bash
Copy code
python LineAdjustment.py
Use the mouse to drag the ROI line to the desired position.
Press:
P to pause/resume the video.
R to reset the ROI line to the default position.
S to save the updated ROI line coordinates.
3. Viewing and Saving Analytics
Analytics data is saved in the following directories:

Grid Counts: grid_counts_csv/
Demographics: demographics_csv/
Rotation intervals ensure CSV files are archived periodically.

Project Structure
bash
Copy code
├── Video_Analytics.py      # Main script for multi-stream object tracking
├── LineAdjustment.py       # Script for dynamic ROI line adjustments
├── requirements.txt        # List of dependencies
├── yolov8n.pt              # YOLOv8 weights for object detection
├── yolov8n-face.pt         # YOLOv8 weights for face detection
├── best_yolov8_model_train24.pt # YOLOv8 weights for gender detection
├── line_coordinates.txt    # ROI line configuration
├── grid_counts_csv/        # Directory for grid-based analytics
├── demographics_csv/       # Directory for demographic insights
Screenshots

Future Enhancements
Heatmap Generation: Visualize customer density in different store areas.
Custom Alerts: Trigger alerts for specific events, like crowd formation.
Edge Deployment: Optimize the system for Jetson devices.
License
This project is licensed under the GPL-3.0 License.


