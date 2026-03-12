
import cv2
from sort import *
import numpy as np
from tkinter import messagebox
import sys
from ultralytics import YOLO
import cvzone
import re
import os
import sys
from tkinter import *
from tkinter import ttk, filedialog
from PIL import Image, ImageTk,ImageEnhance
import threading
from datetime import datetime
import tkinter as tk
import csv
import shutil
import zipfile
import requests
from florence_2_inference import VisionLanguageModel
from phi3_5_inference import TextExtractionModel_Phi_3_Model


# os.environ['API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC']="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"
# api_key = os.getenv("API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC", "")
# if not api_key:
#     print("API_KEY not set. Please export API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC=<Your API Key> as an environment variable.")
#     sys.exit(1)

from dotenv import load_dotenv
import os
import sys

# Load API key from .env file
load_dotenv()  # This reads variables from .env
api_key = os.getenv("API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC", "")

if not api_key:
    print("API_KEY not set. Please create a .env file with API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC=<Your API Key>")
    sys.exit(1)

vlm = VisionLanguageModel(api_key, "https://ai.api.nvidia.com/v1/vlm/microsoft/florence-2")
phi_vlm = TextExtractionModel_Phi_3_Model(api_key, "https://integrate.api.nvidia.com/v1/chat/completions")
# Global variables
running = False
video_path = None
polygon_file_path = None
# Function for intrusion detection

if not os.path.exists("plate_images"):
    os.mkdir("plate_images")

def load_polygon_coordinates(file_path):
    try:
        with open(file_path, 'r') as file:
            line = file.readline().strip()
            # Evaluate the np.array statement from the file
            coordinates = eval(line)
            if isinstance(coordinates, np.ndarray):
                return coordinates
            else:
                raise ValueError("The file does not contain a valid numpy array.")
            
    except Exception as e:
        print(f"Error reading polygon coordinates: {e}")
        return None
    
def run_inference(image_path, output_dir, task_id):
    output_result = vlm.process_task(image_path, output_dir, task_id)

    print("[DEBUG] Raw output_result:", output_result)  # Debugging output

    if isinstance(output_result, dict):  # ✅ Ensure output_result is a dictionary before calling .keys()
        print("[DEBUG] Raw output_result keys:", output_result.keys())  
    else:
        print("[WARNING] output_result is not a dictionary!")

    # ✅ Check if "extracted_data" is missing or empty
    if not output_result or not isinstance(output_result, dict) or not output_result.get("extracted_data"):
        print("[ERROR] extracted_data is missing or empty!")

        print("[WARNING] vlm.process_task() failed or returned an empty result.")
        print("[INFO] Switching to phi_vlm.phi_3_inference()...")

        output_result = phi_vlm.phi_3_inference(image_path)

        print("[DEBUG] Raw output_result from phi_vlm:", output_result)  # Debugging output

        if not output_result or not isinstance(output_result, dict) or "extracted_data" not in output_result:
            print("[ERROR] phi_vlm.phi_3_inference() also failed or returned empty data.")
            output_result = {}

    return output_result

    
# Function to save data as CSV
def save_as_csv(data, headers, filename_suggestion):
    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialfile=filename_suggestion
    )
    if not file_path:
        return  # If the user cancels the save dialog

    # Write data to CSV
    try:
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(headers)  # Write headers
            for row in data:
                writer.writerow(row)  # Write data rows
        messagebox.showinfo("Success", f"Data saved as {file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save CSV: {e}")
    
# Add a dictionary to track detected plates for each car
car_plate_tracker = {}
license_plate_id = set()
license_plate_rows = []

def license_plate_detection_and_recognition():
    global running,car_plate_tracker,license_plate_id,license_plate_rows

    # Check if files are selected
    if  not video_path and not polygon_file_path:
        messagebox.showerror("Error", "Please choose a video file and polygon coordinate txt file")
        return

    if not video_path:
        messagebox.showerror("Error", "Please choose a video file")
        return
    
    if not polygon_file_path:
        messagebox.showerror("Error", "Please choose a polygon coordinate file.")
        return
    
    polygon_coordinates = load_polygon_coordinates(polygon_file_path)

    if polygon_coordinates is not None:
        print("Loaded polygon coordinates:")
        print(polygon_coordinates)
    else:
        print("Failed to load polygon coordinates.")
        messagebox.showerror("Error", "not valid polygon coordinate")


    # Load YOLO models
    car_model = YOLO('yolov8n.pt')  # Car detection model
    plate_model = YOLO('best.pt')  # License plate detection model

    # Initialize video capture and YOLO model
    cap = cv2.VideoCapture(video_path)

    # Load class names
    car_classnames = []
    with open('classes.txt', 'r') as f:
        car_classnames = f.read().splitlines()

    # Initialize SORT tracker
    car_tracker = Sort(max_age=100, min_hits=5, iou_threshold=0.6)

    # Define the original region of interest (polygon region)
    original_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    region_points = polygon_coordinates

    count = 0

    while running:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize the frame for display in Tkinter
        display_width, display_height = 800, 450
        frame = cv2.resize(frame, (display_width, display_height))

        # Scale region points to match the resized frame
        scale_x = display_width / original_frame_width
        scale_y = display_height / original_frame_height
        scaled_region_points = np.array(
            [[int(pt[0] * scale_x), int(pt[1] * scale_y)] for pt in region_points],
            np.int32
        )

        # Draw the polygon region
        # cv2.polylines(frame, [scaled_region_points], isClosed=True, color=(0, 255, 255), thickness=2)

        # Detect cars
        car_detections = car_model(frame, stream=1)
        car_boxes = np.empty((0, 5))
        for result in car_detections:
            boxes = result.boxes
            for box in boxes:
                car_x1, car_y1, car_x2, car_y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                object_detected = car_classnames[int(cls)]
                if object_detected == "car" and conf > 0.5:  # Confidence threshold
                    car_boxes = np.vstack((car_boxes, [car_x1, car_y1, car_x2, car_y2, conf]))

        # Update car tracker
        tracked_cars = car_tracker.update(car_boxes)

        for car in tracked_cars:
            car_x1, car_y1, car_x2, car_y2, car_id = map(int, car)
            car_region = frame[car_y1:car_y2, car_x1:car_x2]

            # Draw the plate bounding box
            cv2.rectangle(frame, (car_x1, car_y1), (car_x2, car_y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Car ID: {car_id}", (car_x1, car_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Draw bounding box and check if the car is inside the polygon
            box_mid = ((car_x1 + car_x2) // 2, (car_y1 + car_y2) // 2)
            if cv2.pointPolygonTest(scaled_region_points, box_mid, False) >= 0:
                if car_id not in car_plate_tracker:
                    car_plate_tracker[car_id] = set()  # Initialize a set to store plates for this car


                # Initialize with an empty list
                plate_detections = []

                # Detect license plates in the car region
                try:
                    plate_detections = plate_model(car_region)
                except:
                    pass
                for result in plate_detections:
                    boxes = result.boxes
                    for box in boxes:
                        px1, py1, px2, py2 = map(int, box.xyxy[0])
                        conf = box.conf[0]
                        if conf > 0.5:  # Confidence threshold
                            # Adjust plate coordinates relative to the original frame
                            abs_px1, abs_py1 = car_x1 + px1, car_y1 + py1
                            abs_px2, abs_py2 = car_x1 + px2, car_y1 + py2
                            
                            # Save plate image only if it hasn't been saved for this car ID
                            plate_key = (abs_px1, abs_py1, abs_px2, abs_py2)
                            if plate_key not in car_plate_tracker[car_id]:
                                car_plate_tracker[car_id].add(plate_key)
                                plate_region = frame[abs_py1:abs_py2, abs_px1:abs_px2]

                                # Convert the OpenCV image (BGR) to a Pillow image (RGB)
                                plate_image = Image.fromarray(cv2.cvtColor(plate_region, cv2.COLOR_BGR2RGB))
                                
                                
                                # 1. Enhance brightness
                                brightness_enhancer = ImageEnhance.Brightness(plate_image)
                                plate_image = brightness_enhancer.enhance(1.2)  # Increase brightness by 20%

                                # 2. Enhance contrast
                                contrast_enhancer = ImageEnhance.Contrast(plate_image)
                                plate_image = contrast_enhancer.enhance(1.4)  # Increase contrast by 30%

                                # 3. Enhance sharpness
                                sharpness_enhancer = ImageEnhance.Sharpness(plate_image)
                                plate_image = sharpness_enhancer.enhance(2.5)  # Double the sharpness

                                if car_id not in license_plate_id:
                                    license_plate_id.add(car_id)  # Add the car_id after the check
                                    plate_image.save(f"plate_images/plat_img_{car_id}_{count}.png")
                                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                                    image_path = f"plate_images/plat_img_{car_id}_{count}.png"
                                    output_dir = "output"
                                    task_id = 13

                                    # Run inference
                                    output_result = run_inference(image_path, output_dir, task_id)

                                    # Debug output
                                    if not output_result or "extracted_data" not in output_result:
                                        print("[WARNING] No extracted data found!")
                                    else:
                                        print("output_result:", type(output_result), output_result["extracted_data"])

                                    log_license_plate_detection(
                                        car_id,
                                        box_mid,
                                        current_time,
                                        output_result["extracted_data"],
                                        f"plate_images/plat_img_{car_id}_{count}.png"
                                    )
                                    license_plate_rows.append([car_id,
                                        box_mid,
                                        current_time,
                                        output_result["extracted_data"],
                                        f"plate_images/plat_img_{car_id}_{count}.png"])

                                count += 1
                                # Draw the plate bounding box
                                cv2.rectangle(frame, (abs_px1, abs_py1), (abs_px2, abs_py2), (255, 0, 0), 2)
                                cv2.putText(frame, f"Plate ID: {car_id}", (abs_px1, abs_py1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Convert frame to ImageTk format for Tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        try:

            imgtk = ImageTk.PhotoImage(image=img)

            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
            root.update()
        except:
            pass

    # destroy all windows
    try:
        cap.release()
        cv2.destroyAllWindows()
    except:
        pass



# Function to start the license plate detection thread
def start_license_plate_detection():
    global running
    running = True
    threading.Thread(target=license_plate_detection_and_recognition).start()

# Function to stop license plate detection
def stop_license_plate_detection():
    global running
    running = False
    sys.exit(1)

# Function to choose video file
def choose_video_file():
    global video_path
    file_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=(
            ("MP4 Files", "*.mp4"),
            ("All Files", "*.*")
        )
    )
    if file_path:
        video_path = file_path

# Function to choose polygon file
def choose_polygon_txt_file():
    global polygon_file_path
    file_path = filedialog.askopenfilename(
        title="Select TXT File with Polygon Coordinates",
        filetypes=(
            ("TXT Files", "*.txt"),
            ("All Files", "*.*")
        )
    )
    if file_path:
        polygon_file_path = file_path

# Function to load and resize images
def load_image(img_path, size=(80, 25)):
    img = Image.open(img_path)
    img = img.resize(size)  # Resize the image
    return ImageTk.PhotoImage(img)

# Function to download an image
def download_image(image_path):
    save_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
    )
    if save_path:
        shutil.copy(image_path, save_path)
        print(f"Image saved to: {save_path}")

# Function to save log as CSV
def save_as_csv(rows, headers, file_name):
    save_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if save_path:
        with open(save_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"Log saved to: {save_path}")

# Function to populate the table
def log_license_plate_detection(obj_id, position, date_time_info,License_plate, license_plate_image_path):
    # Add text information to the Treeview
    row_id = intrusion_list.insert("", "end", values=(obj_id, position, date_time_info,License_plate, license_plate_image_path))
    
    # Add the image to the Image column
    photo = load_image(license_plate_image_path)
    image_refs[row_id] = photo  # Store reference to prevent garbage collection
    Label(image_frame, image=photo, bg="white").grid(row=len(image_refs), column=0, padx=5, pady=5)
    
    # Add a download button for the image with row-specific path
    Button(
        image_frame,
        text="Download",
        command=lambda path=license_plate_image_path: download_image(path)
    ).grid(row=len(image_refs), column=1, padx=5, pady=5)

# Tkinter window setup
root = Tk()
root.title("License Plate Detection System")
root.geometry("1000x600")
root.resizable(True, True)

# # Style configuration
style = ttk.Style()
style.configure("TNotebook.Tab", font=("Arial", 12))
style.configure("Treeview.Heading", font=("Arial", 10, "bold"))
style.configure("Treeview", font=("Arial", 10))

# Tabbed interface
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

# Video tab
video_frame = Frame(notebook, bg="black")
notebook.add(video_frame, text="Live Video")

# License Plate Log tab
log_frame = Frame(notebook)
notebook.add(log_frame, text="License Plate Log")

# Image and button frame
image_frame = Frame(log_frame, bg="white")
image_frame.pack(fill="y", side="right")

# Store references to prevent garbage collection
image_refs = {}


# Video frame setup
Label(
    video_frame,
    text="License Plate Detection and Recognition System",
    font=("Arial", 18, "bold"),
    bg="black", fg="white"
).pack(fill="x", pady=10)

video_label = Label(video_frame, bg="black")
video_label.pack(fill="both", expand=True, padx=10, pady=10)

# Log table setup
columns = ("Object ID", "Position", "Date & Time", "License Plate", "License Plate Image Path")
intrusion_list = ttk.Treeview(log_frame, columns=columns, show="headings", height=15)

for col in columns:
    intrusion_list.heading(col, text=col)
    intrusion_list.column(col, width=150)

intrusion_list.pack(fill="both", expand=True, padx=10, pady=10)

# Download log button
Button(
    log_frame, text="Download License Plate Log as CSV",
    command=lambda: save_as_csv(license_plate_rows, columns, "License_Plate_Log.csv")
).pack(pady=10)


button_frame = Frame(root)
button_frame.pack(fill="x", pady=5)

Button(
    button_frame, text="Choose Video File",
    command=choose_video_file, font=("Arial", 12), bg="blue", fg="white"
).pack(side="left", padx=10)

Button(
    button_frame, text="Choose Polygon Coordinate File",
    command=choose_polygon_txt_file, font=("Arial", 12), bg="blue", fg="white"
).pack(side="left", padx=10)

Button(
    button_frame, text="Start",
    command=start_license_plate_detection, font=("Arial", 12), bg="green", fg="white"
).pack(side="left", padx=10)

Button(
    button_frame, text="Stop",
    command=stop_license_plate_detection, font=("Arial", 12), bg="red", fg="white"
).pack(side="left", padx=10)


# Start Tkinter main loop
root.mainloop()
