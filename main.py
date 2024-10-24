import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import time
from tkinter import Tk, Frame, Label, Canvas, PhotoImage, BOTH

def show_notification(message):
    popup = Toplevel()
    popup.title("Notification")
    popup.geometry("300x100")
    popup.resizable(False, False)
    label = Label(popup, text=message, font=("Helvetica", 12))
    label.pack(pady=20)
    button = Button(popup, text="OK", command=popup.destroy)
    button.pack()

def convert_to_opencv_hsv(h, s, v):
    h = h / 2 if h <= 360 else 179
    s = int(s * 2.55)
    v = int(v * 2.55)
    return (round(int(h)), s, v)

def convert_to_normal_hsv(h, s, v):
    h = round(h * 2)
    s = round(int(s / 2.55))
    v = round(int(v / 2.55))
    return (h, s, v)

color_ranges = {
    'Red1': ([348, 45, 34], [360, 100, 100]),
    'Red2': ([0, 50, 32], [14, 100, 100]),
    'Green': ([63, 15, 20], [175, 100, 100]),
    'Blue1': ([176, 35, 38], [233, 100, 100]),
    'Blue2': ([234, 60, 60], [245, 100, 100]),
    'Yellow': ([40, 40, 32], [62, 100, 100]),
    'Orange': ([15, 75, 75], [40, 100, 100]),
    'Purple': ([250, 30, 20], [290, 100, 100]),
    'White': ([0, 0, 40], [360, 15, 100]),
    'Brown': ([10, 15, 22], [45, 70, 90]),
    'Black': ([0, 0, 0], [360, 100, 12]),
    'Pink': ([300, 20, 40], [340, 100, 100])
}

color_ranges_opencv = {color: [convert_to_opencv_hsv(*lower), convert_to_opencv_hsv(*upper)] for color, (lower, upper) in color_ranges.items()}
bgr = {
    'Red1': (0, 0, 255),
    'Red2': (0, 0, 255),
    'Green': (0, 255, 0),
    'Blue1': (255, 0, 0),
    'Blue2': (255, 0, 0),
    'Yellow': (0, 255, 255),
    'Orange': (0, 165, 255),
    'Purple': (128, 0, 128),
    'White': (255, 255, 255),
    'Brown': (50, 45, 90),
    'Black': (0, 0, 0),
    'Pink': (170, 150, 255)
}

color_order = ['Black', 'Brown', 'White', 'Green', 'Red1', 'Red2', 'Blue1', 'Blue2', 'Yellow', 'Orange', 'Purple', 'Pink']
contour_buffers = {color: [] for color in color_order}
CONTOUR_HISTORY_LENGTH = 1

camera_num = 0
cap = cv2.VideoCapture(camera_num)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.read()


def get_average_hsv(hsv_image, center):
    x, y = center
    samples = []
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if 0 <= x + dx < hsv_image.shape[1] and 0 <= y + dy < hsv_image.shape[0]:
                samples.append(hsv_image[y + dy, x + dx])
    if samples:
        avg_hue = np.mean([sample[0] for sample in samples])
        avg_saturation = np.mean([sample[1] for sample in samples])
        avg_value = np.mean([sample[2] for sample in samples])
        avg_hsv = (int(avg_hue), int(avg_saturation), int(avg_value))
        return avg_hsv
    else:
        return hsv_image[y, x]

def get_hsv_values(image_path):
    image = cv2.imread(image_path)
    width = int(image.shape[1])
    height = int(image.shape[0])
    ar = width / height
    new_width = 4000
    new_height = int(new_width / ar)
    image = cv2.resize(image, (new_width, new_height))
    if image is None:
        show_notification("Error: Could not read the image.")
        return []
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=400, param1=100, param2=30, minRadius=100, maxRadius=450)
    hsv_values = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        if len(circles) < 24:
            print(len(circles))
            show_notification("Detected less than 24 circles, please provide another image.")
            return []
        circles = sorted(circles, key=lambda c: (c[1] // 50, c[0] // 50))
        for i, (x, y, r) in enumerate(circles):
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            hsv_value = get_average_hsv(hsv_image, (x, y))
            if i == 0:  # Adjust the value for the first circle
                hsv_value = (174, hsv_value[1], hsv_value[2])
            if i == 1:  # Adjust the value for the second circle
                hsv_value = (179, hsv_value[1], hsv_value[2])
            if i == 20:  # Adjust the value for the twenty-first circle
                hsv_value = (0, 0, 0)
            if i == 21:  # Adjust the value for the twenty-second circle
                hsv_value = (179, hsv_value[1], hsv_value[2])
            hsv_values.append(hsv_value)
        # Calibration check
        for _ in range(0,len(hsv_values), 2):
            x = hsv_values[_]
            y = hsv_values[_+1]
            condition = True
            if x[0] >= y[0]:
                show_notification("Calibration image provided not accurate, try again")
                condition = False
                hsv_values = []
                break
        if condition: show_notification("Success")
    return hsv_values


def remove_num(color_name):
    return ''.join([char for char in color_name if char.isalpha()])

previous_labels = {}

def process_color(color_name, hsv, output, drawn_contours):
    lower, upper = color_ranges_opencv[color_name]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour_buffers[color_name].append(contours)
        if len(contour_buffers[color_name]) > CONTOUR_HISTORY_LENGTH:
            contour_buffers[color_name].pop(0)
    all_contours = []
    for past_contours in contour_buffers[color_name]:
        all_contours.extend(past_contours)
    if all_contours:
        for contour in all_contours:
            if cv2.contourArea(contour) > 1750:
                x, y, w, h = cv2.boundingRect(contour)
                overlap = drawn_contours[y:y + h, x:x + w].any()
                if not overlap:
                    overlay = output.copy()
                    cv2.drawContours(overlay, [contour], -1, bgr[color_name], -1)
                    cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)
                    drawn_contours[y:y + h, x:x + w] = 1
                    cX, cY = x + w // 2, y + h // 2
                    cv2.putText(output, remove_num(color_name), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    previous_labels[color_name] = (cX - 20, cY, time.time())  # Added timestamp


def release_camera():
    global cap
    if cap:
        cap.release()
        cap = None

def update_frame(label):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (320, 240))
    if not ret:
        return
    blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    output = frame.copy()
    drawn_contours = np.zeros_like(output, dtype="uint8")
    threads = []
    for color_name in color_order:
        thread = threading.Thread(target=process_color, args=(color_name, hsv, output, drawn_contours))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    current_time = time.time()
    for color_name in color_order:
        if color_name in previous_labels:
            cX, cY, label_time = previous_labels[color_name]
            if current_time - label_time < 0.1:  # Show label for only 100 milliseconds
                cv2.putText(output, remove_num(color_name), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    img = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(image=img)
    label.img_tk = img_tk
    label.config(image=img_tk)
    label.after(10, lambda: update_frame(label))

def start_detection():
    release_camera()
    switch_to_frame(video_frame)
    global cap
    cap = cv2.VideoCapture(camera_num)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    update_frame(video_label)

def capture_image():
    release_camera()
    switch_to_frame(capture_frame)
    global cap
    cap = cv2.VideoCapture(camera_num)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    show_camera_feed(capture_label)

def show_camera_feed(label):
    ret, frame = cap.read()
    if not ret:
        return
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (320,240))
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(image=img)
    label.img_tk = img_tk
    label.config(image=img_tk)
    label.after(10, lambda: show_camera_feed(label))

def save_captured_image():
    ret, frame = cap.read()
    if ret:
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            cv2.imwrite(file_path, frame)
    switch_to_frame(home_frame)
    release_camera()

def calibrate_camera():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if file_path:
        hsv_values = get_hsv_values(file_path)
        keys_list = list(color_ranges.keys())
        for i in range(0, len(hsv_values), 2):
            color_ranges[keys_list[i // 2]] = (hsv_values[i], hsv_values[i + 1])

def switch_to_frame(frame):
    for f in (home_frame, capture_frame, video_frame):
        f.pack_forget()
    frame.pack(fill=BOTH, expand=True)

def go_home():
    release_camera()
    switch_to_frame(home_frame)




# Tkinter setup
root = Tk()
root.title("Color Detection App")
root.geometry("360x640")

bg_image = Image.open("background.png")
bg_image = bg_image.resize((360, 640))
bg_photo = ImageTk.PhotoImage(bg_image)

# Home frame
home_frame = Frame(root)
home_frame.pack(fill=BOTH, expand=True)

# Other frames
capture_frame = Frame(root)
video_frame = Frame(root)

# Load images with transparent backgrounds
title_image = PhotoImage(file="ColorView.png")
start_button_image = PhotoImage(file="Start Detecting.png")
capture_button_image = PhotoImage(file="Calibration Capture.png")
save_button_image = PhotoImage(file="Save File.png")
calibrate_button_image = PhotoImage(file="Calibrate.png")
back_button_image = PhotoImage(file="Back.png")

# Function to create a button on a canvas
def create_button(canvas, image, x, y, command):
    button_id = canvas.create_image(x, y, anchor='nw', image=image)
    canvas.tag_bind(button_id, "<Button-1>", lambda e: command())

# Title image
canvas = Canvas(home_frame, width=360, height=640, highlightthickness=0)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, anchor='nw', image=bg_photo)
canvas.create_image(60, 10, anchor='nw', image=title_image)

# Start button
create_button(canvas, start_button_image, 60, 100, start_detection)
# Capture button
create_button(canvas, capture_button_image, 60, 250, capture_image)
# Calibrate button
create_button(canvas, calibrate_button_image, 60, 400, calibrate_camera)

# Frame to show camera feed while capturing
capture_canvas = Canvas(capture_frame, width=480, height=640, highlightthickness=0)
capture_canvas.pack(fill="both", expand=True)
capture_canvas.create_image(0, 0, anchor='nw', image=bg_photo)
capture_label = Label(capture_canvas, width=320, height=240)
capture_label.pack(pady=120)
create_button(capture_canvas, save_button_image, 60, 350, save_captured_image)
create_button(capture_canvas, back_button_image, 60, 400, go_home)

# Video frame with the same background
video_canvas = Canvas(video_frame, width=480, height=640, highlightthickness=0)
video_canvas.pack(fill="both", expand=True)
video_canvas.create_image(0, 0, anchor='nw', image=bg_photo)
video_label = Label(video_canvas, width=320, height=240)
video_label.pack(pady=120)
create_button(video_canvas, back_button_image, 60, 350, go_home)

root.resizable(False, False)
root.mainloop()
