import tkinter as tk
from tkinter import messagebox
from project.utils import Conf
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import os

def train_model():
    try:
        # Load the configuration
        conf = Conf("config/config.json")
        encodings_path = conf["encodings_path"]
        recognizer_path = conf["recognizer_path"]
        le_path = conf["le_path"]

        # Check if encodings file exists
        if not os.path.exists(encodings_path):
            messagebox.showerror("Error", f"Encodings file not found:\n{encodings_path}")
            return

        # Load the face encodings
        print("[INFO] Loading face encodings...")
        with open(encodings_path, "rb") as f:
            data = pickle.load(f)

        if len(data["encodings"]) == 0:
            messagebox.showerror("Error", "No face encodings found. Encode faces first!")
            return

        # Encode the labels
        print("[INFO] Encoding labels...")
        le = LabelEncoder()
        labels = le.fit_transform(data["names"])

        # Train the SVM model
        print("[INFO] Training model...")
        recognizer = SVC(C=1.0, kernel="linear", probability=True)
        recognizer.fit(data["encodings"], labels)

        # Write the model to disk
        os.makedirs(os.path.dirname(recognizer_path), exist_ok=True)
        with open(recognizer_path, "wb") as f:
            pickle.dump(recognizer, f)

        # Write the label encoder to disk
        os.makedirs(os.path.dirname(le_path), exist_ok=True)
        with open(le_path, "wb") as f:
            pickle.dump(le, f)

        # Show success message
        messagebox.showinfo("Success", "Model training completed successfully!")
        exit_program()

    except Exception as e:
        print(e)
        messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

def exit_program():
    root.quit()

# Tkinter window setup
root = tk.Tk()
root.title("Train Face Recognition Model")
root.geometry("500x300")

# Center the window
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = 500
window_height = 300
position_top = int(screen_height / 2 - window_height / 2)
position_left = int(screen_width / 2 - window_width / 2)
root.geometry(f"{window_width}x{window_height}+{position_left}+{position_top}")

# Background color
root.config(bg="#f4f4f9")

# Title label
title_label = tk.Label(root, text="Train Face Recognition Model", font=("Helvetica", 16, "bold"), bg="#f4f4f9")
title_label.pack(pady=10)

# Train button
train_button = tk.Button(root, text="Start Training", command=train_model, font=("Helvetica", 14), bg="#007BFF", fg="white")
train_button.pack(pady=20)

# Exit button
exit_button = tk.Button(root, text="Exit", command=exit_program, font=("Helvetica", 14), bg="#FF4C4C", fg="white")
exit_button.pack(pady=10)

# Run Tkinter main loop
root.mainloop()
