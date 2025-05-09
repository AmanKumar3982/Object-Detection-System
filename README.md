# Object-Detection-System
A real-time object detection web application built with Flask, YOLOv8, and OpenCV. It features a responsive HTML/CSS GUI that allows live video surveillance, object tracking, snapshot capture, and intelligent object summarization. The system also generates automated PDF reports with timestamps using FPDF and sends them via SMTP to the user’s email, all handled in the background using multithreading.

🔧 Features
🎯 Real-time object detection using YOLOv8

🖥️ Live video feed with object tracking via OpenCV

🧠 Smart snapshot capture and object summary generation

📄 Auto-generated PDF reports using FPDF

📬 Email delivery of reports via SMTP

🌐 Responsive HTML/CSS GUI with control buttons

⚙️ Multithreaded background processing for smooth performance

🚀 Technologies Used
Python

Flask

YOLOv8 (Ultralytics)

OpenCV

HTML & CSS

FPDF (for PDF generation)

SMTP (Email handling)

Threading (Background tasks)
object-detection-system/
│
├── static/              # CSS files
├── templates/           # HTML templates
├── snapshots/           # Captured snapshots
├── reports/             # Generated PDF reports
├── app.py               # Main Flask application
├── detector.py          # YOLOv8 detection logic
├── email_utils.py       # Email sending functions
├── requirements.txt     # Required Python packages
└── README.md            # Project description
