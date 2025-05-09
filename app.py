from flask import Flask, render_template, Response, request, send_file
import cv2
from ultralytics import YOLO
import datetime
import os
from fpdf import FPDF
import random
import smtplib
from email.message import EmailMessage
import threading
import time

app = Flask(__name__)




model = YOLO("yolov8n.pt")
detected_objects = set()
user_name = ""
user_email = ""
report_file = "static/detection_report.pdf"
confidence_threshold = 0.6
last_detected_frame = None  # New variable to store snapshot frame


def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))


def send_email_report(to_email, report_path):
    try:
        sender_email = "mail2004amankumar@gmail.com"
        sender_password = "eide lptu tzgx detx"  # Use App Password from Gmail

        msg = EmailMessage()
        msg['Subject'] = 'Your SmartVision Pro Detection Report'
        msg['From'] = sender_email
        msg['To'] = to_email
        msg.set_content('Attached is the object detection report generated from SmartVision Pro.')

        with open(report_path, 'rb') as f:
            file_data = f.read()
            msg.add_attachment(file_data, maintype='application', subtype='pdf', filename='detection_report.pdf')

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)

        print(f"Report sent to {to_email}")
    except Exception as e:
        print(f"Error sending email: {str(e)}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    global user_name
    user_name = request.args.get('username', 'Anonymous')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_frames():
    global detected_objects, last_detected_frame
    cap = cv2.VideoCapture(0)
    object_colors = {}

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False)[0]

        for result in results.boxes:
            if result.conf[0] >= confidence_threshold:
                cls_id = int(result.cls[0])
                label = model.names[cls_id]
                detected_objects.add(label)

                if label not in object_colors:
                    object_colors[label] = random_color()
                color = object_colors[label]

                x1, y1, x2, y2 = map(int, result.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Save the last detected frame
        last_detected_frame = frame.copy()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/detect', methods=['POST'])
def detect():
    global user_name, user_email
    user_name = request.form['username']
    user_email = request.form['email']
    return render_template('success.html', username=user_name, email=user_email)


@app.route('/generate_report')
def generate_report():
    global user_name, user_email, last_detected_frame

    static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

    if not os.path.exists(static_folder):
        os.makedirs(static_folder)

    # Save snapshot image
    snapshot_path = os.path.join(static_folder, "snapshot.jpg")
    if last_detected_frame is not None:
        cv2.imwrite(snapshot_path, last_detected_frame)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt=f"Detection Report for {user_name}", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, txt="Summary of Detected Activities:", ln=True)

    descriptions = []
    person_detected = 'person' in detected_objects

    if person_detected:
        if any(x in detected_objects for x in ['cell phone', 'mobile phone', 'phone']):
            descriptions.append("Person is holding a phone.")
        if 'bottle' in detected_objects:
            descriptions.append("Person is holding a bottle.")
        if 'laptop' in detected_objects:
            descriptions.append("Person is using a laptop.")
        if not any(x in detected_objects for x in ['phone', 'bottle', 'laptop']):
            descriptions.append("A person was detected.")
    else:
        for obj in detected_objects:
            if obj == "cap":
                descriptions.append("Person is wearing a cap.")
            elif obj == "pen":
                descriptions.append("A pen is visible in the scene.")
            elif obj == "pencil":
                descriptions.append("A pencil is visible in the scene.")
            elif obj == "paper":
                descriptions.append("Loose paper is scattered.")
            elif obj == "book":
                descriptions.append("A book is placed in the scene.")
            elif obj == "torch":
                descriptions.append("A torch is lying nearby.")
            elif obj == "mouse":
                descriptions.append("A computer mouse is in use.")
            elif obj == "keyboard":
                descriptions.append("A keyboard is present on the table.")
            elif obj == "cup":
                descriptions.append("A cup is visible.")
            elif obj == "monitor":
                descriptions.append("A monitor screen is present.")
            elif obj == "bag":
                descriptions.append("A bag is kept nearby.")
            elif obj == "backpack":
                descriptions.append("A backpack is visible.")
            elif obj == "chair":
                descriptions.append("A chair is in the scene.")
            elif obj == "table":
                descriptions.append("A table is present.")
            elif obj == "desk":
                descriptions.append("A desk is nearby.")
            elif obj == "glasses":
                descriptions.append("Spectacles are visible.")
            elif obj == "headphones":
                descriptions.append("Headphones are placed nearby.")
            elif obj == "watch":
                descriptions.append("A watch is being worn or placed.")
            elif obj == "wallet":
                descriptions.append("A wallet is visible.")
            elif obj == "broom":
                descriptions.append("A broom is placed in the corner.")
            elif obj == "calculator":
                descriptions.append("A calculator is on the table.")
            elif obj == "scissors":
                descriptions.append("Scissors are placed on the desk.")
            elif obj == "remote":
                descriptions.append("A remote control is present.")
            elif obj == "mouse pad":
                descriptions.append("A mouse pad is visible.")
            elif obj == "tripod":
                descriptions.append("A tripod is set up.")
            elif obj == "ring":
                descriptions.append("A ring is being worn.")
            elif obj == "bracelet":
                descriptions.append("A bracelet is visible.")
            elif obj == "earphones":
                descriptions.append("Earphones are lying nearby.")
            elif obj == "tissue":
                descriptions.append("Tissues or tissue box is present.")
            elif obj == "mask":
                descriptions.append("A face mask is kept.")
            elif obj == "sharpener":
                descriptions.append("A pencil sharpener is in the scene.")
            elif obj == "rubber":
                descriptions.append("An eraser is visible.")
            elif obj == "file":
                descriptions.append("An office file is on the desk.")
            elif obj == "lamp":
                descriptions.append("A desk lamp is visible.")
            elif obj == "highlighter":
                descriptions.append("A highlighter is lying around.")
            elif obj == "ruler":
                descriptions.append("A ruler is placed on the table.")
            elif obj == "sticky notes":
                descriptions.append("Sticky notes are stuck nearby.")
            elif obj == "calendar":
                descriptions.append("A calendar is visible.")
            elif obj == "envelope":
                descriptions.append("An envelope is in the scene.")
            elif obj == "document":
                descriptions.append("Documents are scattered.")
            elif obj == "newspaper":
                descriptions.append("A newspaper is present.")
            elif obj == "magazine":
                descriptions.append("A magazine is visible.")
            elif obj == "tablet":
                descriptions.append("A tablet is in use or nearby.")
            elif obj == "stapler":
                descriptions.append("A stapler is present.")
            elif obj == "tape":
                descriptions.append("A roll of tape is visible.")
            elif obj == "notebook":
                descriptions.append("A notebook is on the desk.")
            elif obj == "screwdriver":
                descriptions.append("A screwdriver is lying around.")
            elif obj == "bottle cap":
                descriptions.append("A bottle cap is on the surface.")
            else:
                descriptions.append(f"{obj.capitalize()} is visible in the scene.")



    if not descriptions:
        descriptions.append("No recognizable objects detected.")

    for line in descriptions:
        pdf.cell(200, 10, txt=f"- {line}", ln=True)

    # Add snapshot to PDF
    if os.path.exists(snapshot_path):
        pdf.ln(10)
        pdf.cell(200, 10, txt="Snapshot of Detection:", ln=True)
        pdf.image(snapshot_path, x=10, y=None, w=180)

    report_path = os.path.join(static_folder, "detection_report.pdf")

    try:
        pdf.output(report_path)
    except Exception as e:
        return f"Error in generating PDF: {str(e)}", 500

    detected_objects.clear()
    last_detected_frame = None

    # Send report via email
    if user_email:
        threading.Thread(target=send_email_report, args=(user_email, report_path)).start()

    # Optional delay
    time.sleep(20)

    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True)
    else:
        return "Report file not found!", 500


if __name__ == '__main__':
    app.run(debug=True)