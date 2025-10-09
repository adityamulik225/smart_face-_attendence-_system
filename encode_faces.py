import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from project.utils import Conf
from imutils import paths
import face_recognition
import pickle
import cv2
import os
import numpy as np

def encode_faces():
    try:
        conf = Conf("config/config.json")
        dataset_path = conf["dataset_path"]
        encodings_path = conf["encodings_path"]

        # Debug: Check dataset folder
        print("Dataset path:", dataset_path)
        print("Subfolders:", os.listdir(dataset_path))

        # Grab all images from subfolders
        imagePaths = list(paths.list_images(dataset_path))
        print("All images found:", imagePaths)

        if len(imagePaths) == 0:
            messagebox.showwarning("Warning", "No images found in the dataset path.")
            return

        knownEncodings = []
        knownNames = []

        progress_bar["maximum"] = len(imagePaths)

        for i, imagePath in enumerate(imagePaths):
            progress_bar["value"] = i + 1
            progress_label.config(text=f"Processing image {i+1}/{len(imagePaths)}")
            root.update_idletasks()

            # Person name from folder
            name = os.path.basename(os.path.dirname(imagePath))

            # Load and convert image
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Encode face
            encodings = face_recognition.face_encodings(rgb)
            if len(encodings) == 0:
                print(f"⚠️ No face found in {imagePath}, skipping...")
                continue

            for encoding in encodings:
                knownEncodings.append(encoding)
                knownNames.append(name)

        # Save encodings
        data = {"encodings": knownEncodings, "names": knownNames}
        os.makedirs(os.path.dirname(encodings_path), exist_ok=True)
        with open(encodings_path, "wb") as f:
            pickle.dump(data, f)

        messagebox.showinfo("Success", f"Encoding completed! {len(imagePaths)} images processed.")
        exit_program()

    except Exception as e:
        messagebox.showerror("Error", str(e))

def exit_program():
    root.quit()

# Tkinter UI setup
root = tk.Tk()
root.title("Face Encoder")
root.geometry("500x300")

window_width = 500
window_height = 300
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
position_top = int(screen_height/2 - window_height/2)
position_left = int(screen_width/2 - window_width/2)
root.geometry(f"{window_width}x{window_height}+{position_left}+{position_top}")
root.config(bg="#f4f4f9")

title_label = tk.Label(root, text="Face Encoding", font=("Helvetica",16,"bold"), bg="#f4f4f9")
title_label.pack(pady=10)

progress_bar = ttk.Progressbar(root, length=400, mode="determinate")
progress_bar.pack(pady=20)

progress_label = tk.Label(root, text="Waiting to start ...", font=("Helvetica",12), bg="#f4f4f9")
progress_label.pack()

encode_button = tk.Button(root, text="Start Encoding", command=encode_faces, font=("Helvetica",14), bg="#007BFF", fg="white")
encode_button.pack(pady=20)

exit_button = tk.Button(root, text="Exit", command=exit_program, font=("Helvetica",14), bg="#FF4C4C", fg="white")
exit_button.pack(pady=10)

root.mainloop()
