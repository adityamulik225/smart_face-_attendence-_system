import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from datetime import datetime, time
import pickle
import json
from tinydb import TinyDB
import face_recognition
from project.utils import Conf
import os
from pymongo import MongoClient
import threading
import pyttsx3  

conf = Conf("config/config.json")
recognizer = pickle.loads(open(conf["recognizer_path"], "rb").read())
le = pickle.loads(open(conf["le_path"], "rb").read())


client = MongoClient("mongodb+srv://smartmess:smartmessdev2025@cluster0.vhgy1fm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["test"]
attendance_collection = db["AttendanceData"]


db_local = TinyDB(conf["db_path"])
studentTable = db_local.table("student")


vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
vs.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)


today_date = datetime.now().strftime("%Y-%m-%d")
json_file_path_attendance = f"attendance_{today_date}.json"

if not os.path.exists(json_file_path_attendance):
    with open(json_file_path_attendance, 'w') as f:
        json.dump({}, f, indent=4)

try:
    with open(json_file_path_attendance, 'r') as f:
        attendance_cache = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    attendance_cache = {}


def get_meal_mode(current_time=None):
    if current_time is None:
        current_time = datetime.now().time()
    lunch_start, lunch_end = time(11, 45), time(14, 30)
    dinner_start, dinner_end = time(19, 0), time(21, 30)

    if lunch_start <= current_time <= lunch_end:
        return "Lunch"
    elif dinner_start <= current_time <= dinner_end:
        return "Dinner"
    else:
        return "General"


voice_lock = threading.Lock()
engine = pyttsx3.init()
engine.setProperty('rate', 180)
engine.setProperty('volume', 1.0)

def play_unknown_alert():
    def speak():
        with voice_lock:
            engine.say("Unknown face detected. Please try again.")
            engine.runAndWait()
    threading.Thread(target=speak, daemon=True).start()


def store_attendance(name, id_):
    if not name or name.lower() == "unknown":
        play_unknown_alert()
        alert_label.config(text="⚠️ Unknown face detected!", fg="red")
        root.after(2000, lambda: alert_label.config(text=""))
        return "Unknown face detected — not stored."

    now = datetime.now()
    current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    mode = get_meal_mode(now.time())

    if "attendance" not in attendance_cache:
        attendance_cache.clear()
        attendance_cache["date_time"] = current_time_str
        attendance_cache["mode"] = mode
        attendance_cache["attendance"] = []

    if attendance_cache.get("mode") != mode:
        attendance_cache.clear()
        attendance_cache["date_time"] = current_time_str
        attendance_cache["mode"] = mode
        attendance_cache["attendance"] = []

    for record in attendance_cache["attendance"]:
        if record["id"] == id_:
            return f"Attendance already recorded for {name} ({mode})."

    attendance_cache["attendance"].append({
        "id": id_,
        "name": name
    })

    ordered_data = {
        "date_time": attendance_cache["date_time"],
        "mode": attendance_cache["mode"],
        "attendance": attendance_cache["attendance"]
    }

    with open(json_file_path_attendance, 'w') as f:
        json.dump(ordered_data, f, indent=4)

    alert_label.config(text=f"✔ Attendance stored for {name} ({mode})", fg="green")
    root.after(2000, lambda: alert_label.config(text=""))
    return f"Attendance stored for {name} ({mode})"


def save_to_mongodb():
    today_date = datetime.now().strftime("%Y-%m-%d")
    if "attendance" not in attendance_cache or not attendance_cache["attendance"]:
        messagebox.showwarning("No Data", "No attendance data available to save!")
        return

    existing_doc = attendance_collection.find_one({"date": today_date})
    students_list = []
    existing_records = set()

    if existing_doc:
        existing_records = {s["ID"] for s in existing_doc["students"]}

    for record in attendance_cache["attendance"]:
        if record["id"] not in existing_records:
            students_list.append({
                "ID": record["id"],
                "Name": record["name"],
                "Mode": attendance_cache.get("mode", "Unknown"),
                "DateTime": attendance_cache.get("date_time", "")
            })

    if not students_list:
        messagebox.showinfo("Info", "All today's records already exist in MongoDB.")
        return

    if existing_doc:
        attendance_collection.update_one(
            {"date": today_date},
            {"$push": {"students": {"$each": students_list}}}
        )
    else:
        attendance_collection.insert_one({
            "date": today_date,
            "students": students_list
        })

    messagebox.showinfo("Success", f"✔ {len(students_list)} new records added for {today_date}.")


root = tk.Tk()
root.title("Smart Face Attendance System")
root.geometry("850x720")

attendance_label = tk.Label(root, text="Attendance Recognition: Ready", font=("Arial", 16))
attendance_label.pack(pady=10)

alert_label = tk.Label(root, text="", font=("Arial", 14))
alert_label.pack(pady=5)

canvas = tk.Canvas(root, width=640, height=480)
canvas.pack(pady=10)

video_running = False
FRAME_SKIP = 8
frame_count = 0
last_boxes = []
last_names = []


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

        boxes_small = face_recognition.face_locations(rgb_small)
        encodings = face_recognition.face_encodings(rgb_small, boxes_small)

        last_boxes = []
        last_names = []

        detected_faces = []   

        for encoding, box in zip(encodings, boxes_small):
            preds = recognizer.predict_proba([encoding])[0]
            j = np.argmax(preds)
            confidence = preds[j]

            if confidence < 0.8:
                name, curPerson = "Unknown", "Unknown"
            else:
                curPerson = le.classes_[j]
                result = studentTable.search(lambda doc: curPerson in doc)
                name = result[0][curPerson][0] if result else "Unknown"

            detected_faces.append((name, curPerson, box))

   
        for (name, curPerson, box) in detected_faces:
            attn_info = store_attendance(name, curPerson)
            attendance_label.config(text=f"Status: {attn_info}")

            top, right, bottom, left = [v * 2 for v in box]
            last_boxes.append((top, right, bottom, left))
            last_names.append(name)

    for (top, right, bottom, left), name in zip(last_boxes, last_names):
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = img.resize((640, 480), Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image = img_tk

    root.after(50, update_frame)


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


button_frame = tk.Frame(root)
button_frame.pack(pady=15)

start_button = tk.Button(button_frame, text="Start", font=("Arial", 16), bg="#00cc66", command=start_video)
start_button.grid(row=0, column=0, padx=10)

save_button = tk.Button(button_frame, text="Save to Database", font=("Arial", 16), bg="#007bff", fg="white", command=save_to_mongodb)
save_button.grid(row=0, column=1, padx=10)

exit_button = tk.Button(button_frame, text="Exit", font=("Arial", 16), bg="#ff3333", command=exit_program)
exit_button.grid(row=0, column=2, padx=10)

root.mainloop()

vs.release()
cv2.destroyAllWindows()
