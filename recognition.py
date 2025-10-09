import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from datetime import datetime
import pickle
import json
from tinydb import TinyDB
import face_recognition
from project.utils import Conf
import pandas as pd
import os

# ---------------------- Load Configuration and Models ----------------------
conf = Conf("config/config.json")
recognizer = pickle.loads(open(conf["recognizer_path"], "rb").read())
le = pickle.loads(open(conf["le_path"], "rb").read())

# ---------------------- TinyDB and JSON Paths ----------------------
db = TinyDB(conf["db_path"])
studentTable = db.table("student")
json_file_path_attendance = 'attendance.json'

# ---------------------- Video Capture ----------------------
vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
vs.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# ---------------------- Attendance cache ----------------------
attendance_cache = {"attendance": {}}

# Load existing attendance
try:
    with open(json_file_path_attendance, 'r') as f:
        attendance_cache = json.load(f)
        if "attendance" not in attendance_cache:
            attendance_cache = {"attendance": {}}
except (FileNotFoundError, json.JSONDecodeError):
    attendance_cache = {"attendance": {}}

# ---------------------- Attendance Functions ----------------------
def store_attendance(name, id_):
    if not name or name.lower() == "unknown":
        return None

    today_date = datetime.now().strftime("%Y-%m-%d")

    # Reset if day changed
    last_dates = [v['date_time'].split(" ")[0] for v in attendance_cache['attendance'].values()]
    last_date = last_dates[0] if last_dates else today_date
    if last_date != today_date:
        attendance_cache['attendance'] = {}

    # Already recorded today?
    if id_ in attendance_cache['attendance']:
        recorded_date = attendance_cache['attendance'][id_]['date_time'].split(" ")[0]
        if recorded_date == today_date:
            return f"Attendance already recorded for {name} today."

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    attendance_cache['attendance'][id_] = {"name": name, "date_time": current_time}

    # Save to JSON backup
    with open(json_file_path_attendance, 'w') as f:
        json.dump(attendance_cache, f, indent=4)

    # --- Save to Excel on Desktop ---
    rows = [{'ID': k, 'Name': v['name'], 'DateTime': v['date_time']} 
            for k, v in attendance_cache['attendance'].items()]

    if rows:
        df = pd.DataFrame(rows)
        folder_path = r"C:\Users\Admin\OneDrive\Desktop\attendence"  # <-- Corrected path
        os.makedirs(folder_path, exist_ok=True)

        file_name = f"{today_date}_Attendance.xlsx"
        file_path = os.path.join(folder_path, file_name)
        df.to_excel(file_path, index=False)

    return f"Attendance stored for {name}"

# ---------------------- Tkinter GUI ----------------------
root = tk.Tk()
root.title("Smart Face Attendance System")
root.geometry("800x700")

attendance_label = tk.Label(root, text="Attendance Recognition: Ready", font=("Arial", 16))
attendance_label.pack(pady=20)

canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

video_running = False
FRAME_SKIP = 8  # Detect every 8 frames
frame_count = 0
last_boxes = []
last_names = []

# ---------------------- Frame Update ----------------------
def update_frame():
    global video_running, frame_count, last_boxes, last_names
    if not video_running:
        return

    ret, frame = vs.read()
    if not ret:
        root.after(50, update_frame)
        return

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    frame_count += 1
    if frame_count % FRAME_SKIP == 0:
        boxes_small = face_recognition.face_locations(rgb_small, model=conf["detection_method"])
        encodings = face_recognition.face_encodings(rgb_small, boxes_small)

        last_boxes = []
        last_names = []

        for encoding, box in zip(encodings, boxes_small):
            preds = recognizer.predict_proba([encoding])[0]
            j = np.argmax(preds)
            curPerson = le.classes_[j]

            # Lookup student name
            result = studentTable.search(lambda doc: curPerson in doc)
            name = result[0][curPerson][0] if result else "Unknown"

            attn_info = store_attendance(name, curPerson)
            if attn_info:
                attendance_label.config(text=f"Attendance Status: {attn_info}")

            # Scale box to original frame
            top, right, bottom, left = box
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            last_boxes.append((top, right, bottom, left))
            last_names.append(name)

    # Draw rectangles and names
    for (top, right, bottom, left), name in zip(last_boxes, last_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Convert frame to Tkinter image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if hasattr(Image, 'Resampling'):
        img = img.resize((640, 480), Image.Resampling.LANCZOS)
    else:
        img = img.resize((640, 480), Image.ANTIALIAS)

    img_tk = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image = img_tk

    root.after(50, update_frame)

# ---------------------- Buttons ----------------------
def start_video():
    global video_running
    video_running = True
    update_frame()

def exit_program():
    global video_running
    video_running = False
    vs.release()
    cv2.destroyAllWindows()
    root.quit()

start_button = tk.Button(root, text="Start", font=("Arial", 16), bg="#00ff00", command=start_video)
start_button.pack(pady=10)

exit_button = tk.Button(root, text="Exit", font=("Arial", 16), bg="#ff0000", command=exit_program)
exit_button.pack(pady=10)

# ---------------------- Run Tkinter ----------------------
root.mainloop()

# Cleanup
vs.release()
cv2.destroyAllWindows()
