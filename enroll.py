import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from project.utils import Conf
from tinydb import TinyDB 
import face_recognition
import cv2
import os 
import threading

# Create required directories if not exist
if not os.path.exists("dataset"):
    os.mkdir("dataset")
if not os.path.exists("dataset/PROJECT"):
    os.mkdir("dataset/PROJECT")   

# Stop event to indicate if enrollment should be stopped
stop_event = threading.Event()

# Function to handle enrollment process
def enroll_student():
    stop_event.clear()  # Reset stop event each time process starts
    enroll_button.config(state=tk.DISABLED)  # Disable enroll button to prevent multiple submissions

    # Retrieve input values
    student_id = entry_id.get().strip()
    student_name = entry_name.get().strip()
    config_file = config_path.get().strip()

    # Validate input
    if not student_id or not student_name:
        messagebox.showerror("Input Error","Please provide Name and ID")
        enroll_button.config(state=tk.NORMAL)
        return
    if not student_id.isdigit():
        messagebox.showerror("Input Error","Please provide valid numeric ID")
        enroll_button.config(state=tk.NORMAL)
        return
    if not os.path.exists(config_file):
        messagebox.showerror("File Error", f"Config file '{config_file}' does not exist.")
        enroll_button.config(state=tk.NORMAL)
        return

    # Load configuration
    conf = Conf(config_file)

    # Initialize database
    db = TinyDB(conf["db_path"])
    student_table = db.table("student")

    # Check if student is already enrolled
    for record in student_table.all():
        for sub_key in record:
            if student_id == sub_key:
                messagebox.showinfo("Already Enrolled", f"Person ID: '{student_id}' is already enrolled.")
                db.close()
                enroll_button.config(state=tk.NORMAL)
                return

    # Thread for face enrollment
    def process_enrollment():
        try:
            vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            student_path = os.path.join(conf["dataset_path"], conf["class_name"], student_id)
            os.makedirs(student_path, exist_ok=True)

            total_saved = 0
            while total_saved < conf["face_count"]:
                if stop_event.is_set():
                    messagebox.showinfo("Process Stopped", "Enrollment process has been stopped.")
                    break

                ret, frame = vs.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes = face_recognition.face_locations(rgb_frame, model=conf["detection_method"])
                frame_copy = frame.copy()

                # Draw boxes and save faces
                for (top, right, bottom, left) in boxes:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)

                    padding = 70
                    top = max(0, top - padding)
                    bottom = min(frame.shape[0], bottom + padding)
                    left = max(0, left - padding)
                    right = min(frame.shape[1], right + padding)
                    face_image = frame_copy[top:bottom, left:right]
                    if total_saved < conf["face_count"]:
                        save_path = os.path.join(student_path, f"{str(total_saved).zfill(5)}.png")
                        cv2.imwrite(save_path, face_image)
                        total_saved += 1
                        root.after(0, update_progress, total_saved, conf["face_count"])

                cv2.putText(frame, "Status: Saving", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow("Frame", frame)
                cv2.waitKey(1)

            vs.release()
            cv2.destroyAllWindows()

            if not stop_event.is_set():
                student_table.insert({student_id: [student_name, "enrolled"]})
                messagebox.showinfo("Success", f"Enrollment completed for {student_name}.")
                reset_form()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            db.close()
            enroll_button.config(state=tk.NORMAL)

    threading.Thread(target=process_enrollment, daemon=True).start()

# Exit program
def exit_program():
    root.quit()

# Update progress bar
def update_progress(total_saved, total_faces):
    progress_bar["value"] = (total_saved / total_faces) * 100
    percentage_label.config(text=f"{int((total_saved / total_faces) * 100)}%")

# Stop enrollment process
def stop_enrollment():
    stop_event.set()
    messagebox.showinfo("Stopping", "Stopping the enrollment process.")

# Reset form
def reset_form():
    entry_id.delete(0, tk.END)
    entry_name.delete(0, tk.END)
    progress_bar["value"] = 0
    percentage_label.config(text="0%")
    messagebox.showinfo("Reset", "Form cleared.")

# Browse config file
def browse_config():
    file_path = filedialog.askopenfilename(title="Select Config File", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
    if file_path:
        config_path.config(state=tk.NORMAL)
        config_path.delete(0, tk.END)
        config_path.insert(0, file_path)
        config_path.config(state=tk.DISABLED)

# Gradient background
def draw_gradient(canvas, width, height):
    canvas.delete("gradient")
    for i in range(256):
        color = f"#{int(0.6*i):02x}{int(0.8*i):02x}{i:02x}"
        y1 = int(i * height / 256)
        y2 = int((i+1) * height / 256)
        canvas.create_rectangle(0, y1, width, y2, fill=color, outline="", tags="gradient")

# Tkinter window
root = tk.Tk()
root.title("Face Enrollment")
root.geometry("800x600")
root.configure(bg="#eef2f3")

gradient_canvas = tk.Canvas(root, highlightthickness=0)
gradient_canvas.pack(fill="both", expand=True)
gradient_canvas.bind("<Configure>", lambda e: draw_gradient(gradient_canvas, e.width, e.height))

title_label = tk.Label(root, text="Face Enrollment", font=("Helvetica", 22, "bold"), bg="#ff0000", fg="white")
title_label.place(relx=0.5, rely=0.05, anchor="n", width=400)

input_frame = tk.Frame(root, bg="#ffffff", padx=20, pady=20, relief="solid", bd=2)
input_frame.place(relx=0.5, rely=0.4, anchor="center", relwidth=0.6, relheight=0.5)

def create_labeled_entry(parent, label_text, default=""):
    label = tk.Label(parent, text=label_text, font=("Helvetica", 12), bg="#ffffff")
    label.pack(anchor="w", pady=5)
    entry = tk.Entry(parent, font=("Helvetica", 14))
    entry.insert(0, default)
    entry.pack(fill="x", pady=5)
    return entry

entry_id = create_labeled_entry(input_frame, "Person ID:")
entry_name = create_labeled_entry(input_frame, "Person Name:")
config_path = create_labeled_entry(input_frame, "Config Path:", default="config/config.json")
config_path.config(state=tk.DISABLED)

browse_button = tk.Button(input_frame, text="Browse", command=browse_config, font=("Helvetica", 12))
browse_button.pack(pady=5)

progress_bar = ttk.Progressbar(input_frame, length=300, mode="determinate")
progress_bar.pack(pady=20)
percentage_label = tk.Label(input_frame, text="0%", font=("Helvetica", 14), bg="#ffffff")
percentage_label.pack(pady=10)

button_frame = tk.Frame(root, bg="#eef2f3")
button_frame.place(relx=0.5, rely=0.8, anchor="center")

enroll_button = tk.Button(button_frame, text="Enroll", font=("Helvetica", 14, "bold"), bg="#ff0000", fg="white", command=enroll_student)
enroll_button.pack(side=tk.LEFT, padx=10)
stop_button = tk.Button(button_frame, text="Stop Enrollment", font=("Helvetica", 14, "bold"), bg="#ff0000", fg="white", command=stop_enrollment)
stop_button.pack(side=tk.LEFT, padx=10)
reset_button = tk.Button(button_frame, text="Reset", font=("Helvetica", 14, "bold"), bg="#ff0000", fg="white", command=reset_form)
reset_button.pack(side=tk.LEFT, padx=10)
exit_button = tk.Button(root, text="Exit", command=exit_program, font=("Helvetica", 14), bg="#ff0000", fg="white")
exit_button.pack(pady=10)

style = ttk.Style(root)
style.theme_use("default")
style.configure("TProgressbar", troughcolor="#e0e0e0", background="#ff0000", thickness=20)

root.mainloop()
