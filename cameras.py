import cv2
from ultralytics import YOLO
import threading
import numpy as np
import pygame
import os
import shutil
import subprocess
import time
import csv

image_folder = os.path.abspath("images")
output_folder = os.path.abspath("output")

# Cache for debugging purposes, will be removed once photogrammetry is completed.
cache_folder = os.path.abspath("cache")
os.makedirs(cache_folder, exist_ok=True)
pipeline = 'photogrammetryDraft'
ENABLE_LOGGING = False
interrupt = False

# Remove old dirs (if present) then recreate empty
for p in (image_folder, output_folder):
    if os.path.exists(p):
        try:
            shutil.rmtree(p)
        except Exception as e:
            print(f"Failed to remove {p}: {e}")
    os.makedirs(p, exist_ok=True)

model = YOLO("best.pt")

# Camera Setup

class Frame:
    def __init__(self):
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.lock = threading.Lock()
        self.last_update = time.time()
        self.latency = time.time()

    def set(self, frame, start_time):
        with self.lock:
            self.frame = frame.copy()
            self.latency = start_time

    def get(self):
        with self.lock:
            return self.frame.copy(), self.latency

def telemetry_logger(frame_list, filename="vision_performance.csv"):
    print(f"[Telemetry] Logging started. Saving to {filename}")
    
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Cam0_Latency", "Cam1_Latency", "Cam2_Latency", "Cam3_Latency"])

    while not interrupt:
        time.sleep(10)
        timestamp = time.strftime("%H:%M:%S")
        now = time.time()
        current_latencies = [now - f.get()[1] for f in frame_list]
        
        with open(filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp] + current_latencies)
            
    print("[Telemetry] Logging stopped.")

def combine(imgs):
    img1 = cv2.resize(imgs[0], (1920, 1080))
    img2 = cv2.resize(imgs[1], (640, 360))
    img3 = cv2.resize(imgs[2], (640, 360))
    img4 = cv2.resize(imgs[3], (640, 360))
    img5 = cv2.hconcat([img2, img3, img4])
    return cv2.vconcat([img1, img5])

def run_camera(url, frame_obj, model=None, camera_id=0):
    global interrupt
    retry_count = 0

    while not interrupt:
        print(f"[Executive] Initializing Stream {camera_id}...")

        wait_time = min(retry_count * 2, 10) 
        if retry_count > 0:
            print(f"[Executive] Retry {retry_count} for Stream {camera_id} in {wait_time}s...")
            time.sleep(wait_time)

        video = cv2.VideoCapture(url)
        
        last_heartbeat = time.time()
        timeout_threshold = 2.0  # If no frames for 2 seconds, it's a "Stall"

        while not interrupt:
            start_time = time.time() if ENABLE_LOGGING else 0
            r, f = video.read()
            
            if r and f is not None:
                last_heartbeat = time.time()
                
                if model and camera_id == 0:
                    results = model.predict(f, verbose=False)
                    f = results[0].plot()

                    # Count Crabs on Upper Left Corner
                    number = len(results[0].boxes)
                    cv2.putText(f, f"Green Crabs Detected: {number}", (7, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                
                frame_obj.set(f, start_time)

            if (time.time() - last_heartbeat) > timeout_threshold:
                print(f"[Watchdog] Stream {camera_id} HEARTBEAT LOST. Auto-Recovering...")
                break 
                
        video.release()
        time.sleep(1) 
        retry_count += 1

# Photogrammetry & Controller
photogrammetryProc = None
controller = None
generating = False
numPictures = 0

def connect_controller():
    global controller
    if pygame.joystick.get_count() > 0 and controller is None:
        controller = pygame.joystick.Joystick(0)
        controller.init()
        print(f'[System] Controller connected: {controller.get_name()}')
    elif pygame.joystick.get_count() == 0 and controller is not None:
        print('[System] Controller disconnected')
        controller = None

def run_photogrammetry():
    global generating, photogrammetryProc
    generating = True

    # Get absolute paths (Required for Docker on macOS)
    base_path = os.path.dirname(os.path.abspath(__file__))
    pipeline_path = os.path.join(base_path, "draft_pipeline.mg")

    if not os.path.isfile(pipeline_path):
        print(f"[ERROR] Could not find pipeline file at: {pipeline_path}")
        return

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{image_folder}:/input",
        "-v", f"{output_folder}:/output",
        "-v", f"{pipeline_path}:/pipeline.mg", 
        "-v", f"{cache_folder}:/tmp/MeshroomCache",
        "alicevision/meshroom:2025.1.0-av3.3.0-ubuntu22.04-cuda12.1.1",
        "/opt/Meshroom_bundle/meshroom_batch", 
        "--input", "/input", 
        "--output", "/output", 
        "--pipeline", "/pipeline.mg" 
    ]

    with open("photogrammetry.log", "w") as logfile:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        photogrammetryProc = proc

        for line in proc.stdout:
            print(line, end="")      # terminal
            logfile.write(line)      # file
        ret = proc.wait()

    photogrammetryProc = None
    if ret == 0:
        print("[System] PHOTOGRAMMETRY FINISHED")
    elif ret != -9:
        print(f"[System] Photogrammetry exited with code {ret}")

    generating = False

# Threads
frames = []
threads = []

# Change depending on amount of cameras available
urls = [
    "udp://192.168.2.1:50000?fifo_size=1000000&overrun_nonfatal=1",
    "udp://192.168.2.1:50001?fifo_size=1000000&overrun_nonfatal=1",
    "udp://192.168.2.1:50002?fifo_size=1000000&overrun_nonfatal=1",
    "udp://192.168.2.1:50002?fifo_size=1000000&overrun_nonfatal=1",
]

for i in range(len(urls)):
    frames.append(Frame())
    target_model = model if i == 0 else None
    threads.append(threading.Thread(target=run_camera, args=(urls[i], frames[i], target_model, i)))
    threads[i].daemon = True 
    threads[i].start()

log_thread = None
if ENABLE_LOGGING:
    log_thread = threading.Thread(target=telemetry_logger, args=(frames,))
    log_thread.daemon = True
    log_thread.start()

print("[System] All Vision Threads Active.")

# Set up UI / Controllers
pygame.init()
pygame.joystick.init()
connect_controller()
if controller is None:
    print("[System] No controller connected. Plug in a controller to use photogrammetry")

pictureWasPressed = True
generateWasPressed = True
photogrammetryThread = None

# Main Loop
try:
    while not interrupt:
        raw_data = [f.get() for f in frames]
        imgs = [data[0] for data in raw_data]

        if len(imgs) == 4:
            combined = combine(imgs)
            
            # Latency Overlay 
            if ENABLE_LOGGING: 
                current_latencies = [time.time() - data[1] for data in raw_data]
                cv2.putText(combined, f"Latency: {current_latencies[0]:.3f}s", (7, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Slugbotics Topside', combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            interrupt = True
            
        pygame.event.pump()
        connect_controller()
        if controller is not None:
            picture = bool(controller.get_button(0))   # A Button
            generate = bool(controller.get_button(1))  # B Button
            
            if picture and not pictureWasPressed:
                print(f"[System] Image saved in images/img{numPictures}.png")
                filename = os.path.join(image_folder, f'img{numPictures}.jpg')

                # Jpeg for Meshroom
                success = cv2.imwrite(filename, imgs[0], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            
                if success:
                    print(f"[System] Image successfully saved: {filename}")
                else:
                    print(f"[ERROR] Failed to save image to: {filename}. Check folder permissions.")
                numPictures += 1
                
            if generate and not generateWasPressed:
                if generating:
                    print('[System] Photogrammetry is already running')
                else:
                    photogrammetryThread = threading.Thread(target=run_photogrammetry)
                    photogrammetryThread.start()
                    
            pictureWasPressed, generateWasPressed = picture, generate

except KeyboardInterrupt:
    print("\n[System] User-initiated interrupt (Ctrl+C). Shutting down...")

except Exception as e:
    print(f'[CRITICAL ERROR] {e}')

# Cleanup & Shutdown
finally: 
    interrupt = True
    print("[System] Shutdown signaled. Beginning exit.")

    for i, t in enumerate(threads):
        t.join(timeout=2.0) 
        print(f"[System] Video Stream {i} released.")

    if photogrammetryThread is not None and photogrammetryThread.is_alive():
        if photogrammetryProc:
            print("[System] Killing background photogrammetry process...")
            photogrammetryProc.kill()
        photogrammetryThread.join(timeout=2.0)

    if log_thread:
        log_thread.join(timeout=2.0)
        print("[System] Telemetry data flushed to disk.")

    cv2.destroyAllWindows()
    pygame.quit()
    print("[System] Exit complete.\n")