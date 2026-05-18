import cv2
from ultralytics import YOLO
import threading
import numpy as np
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
ENABLE_LOGGING = True
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

# CAMERA SETUP
class Frame:
    def __init__(self):
        self.raw_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.processed_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.lock = threading.Lock()
        self.latency = time.time()
        self.inference_time = 0.0

    def set_raw(self, frame, start_time):
        with self.lock:
            self.raw_frame = frame.copy()
            self.latency = start_time

    def set_processed(self, frame, inf_time):
        with self.lock:
            self.processed_frame = frame.copy()
            self.inference_time = inf_time

    def get(self):
        with self.lock:
            return self.processed_frame.copy(), self.latency, self.inference_time

    def get_raw(self):
        with self.lock:
            return self.raw_frame.copy(), self.latency

def telemetry_logger(frame_list, filename="vision_performance.csv", recovery_file="recovery_performance.csv"):
    if not ENABLE_LOGGING: return
    
    with open(filename, mode='w', newline='') as f:
        csv.writer(f).writerow(["Timestamp", "Total_AI_Stream_Lat", "Pure_AI_Inference", "Raw_1", "Raw_2", "Raw_3"])
    with open(recovery_file, mode='w', newline='') as f:
        csv.writer(f).writerow(["Timestamp", "Camera_ID", "Recovery_Duration_Sec"])

    while not interrupt:
        time.sleep(10)
        timestamp = time.strftime("%H:%M:%S")
        now = time.time()
        
        data = [f.get() for f in frame_list]
        total_latencies = [now - d[1] for d in data]
        pure_inf = data[0][2]  # Only Cam0 has inference time 
        
        row = [timestamp, total_latencies[0], pure_inf] + total_latencies[1:]
        
        with open(filename, mode='a', newline='') as f:
            csv.writer(f).writerow(row)

def combine(imgs):
    img1 = cv2.resize(imgs[0], (1920, 1080))
    img2 = cv2.resize(imgs[1], (640, 360))
    img3 = cv2.resize(imgs[2], (640, 360))
    img4 = cv2.resize(imgs[3], (640, 360))
    img5 = cv2.hconcat([img2, img3, img4])
    return cv2.vconcat([img1, img5])

def run_inference(model, frame_obj):
    global interrupt
    last_processed_time = 0
    
    while not interrupt:
        f, frame_time = frame_obj.get_raw() 
        
        if f is not None and np.any(f) and frame_time != last_processed_time:
            last_processed_time = frame_time 
            
            inf_start = time.time()
            results = model.predict(f, verbose=False)
            inf_duration = time.time() - inf_start
            
            annotated_frame = results[0].plot()
            number = len(results[0].boxes)
            cv2.putText(annotated_frame, f"Green Crabs: {number}", (7, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            
            frame_obj.set_processed(annotated_frame, inf_duration)
        else:
            time.sleep(0.005)

def run_camera(url, frame_obj, camera_id=0):
    global interrupt
    retry_count = 0

    while not interrupt:
        print(f"[Executive] Initializing Stream {camera_id}...")
        wait_time = min(retry_count * 2, 10) 
        if retry_count > 0:
            print(f"[Executive] Retry {retry_count} for Stream {camera_id} in {wait_time}s...")
            time.sleep(wait_time)

        recovery_start = time.time() if ENABLE_LOGGING else 0
        video = cv2.VideoCapture(url)
        video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        last_heartbeat = time.time()
        timeout_threshold = 2.0 # If no frames for 2 seconds, it's a "Stall"
        first_frame_received = False

        while not interrupt:
            start_time = time.time() if ENABLE_LOGGING else 0
            r, f = video.read()
            
            if r and f is not None:
                if ENABLE_LOGGING and not first_frame_received:
                    duration = time.time() - recovery_start
                    with open("recovery_performance.csv", mode='a', newline='') as rf:
                        csv.writer(rf).writerow([time.strftime("%H:%M:%S"), camera_id, f"{duration:.3f}"])
                    first_frame_received = True

                frame_obj.set_raw(f, start_time)
                if camera_id != 0:
                    frame_obj.set_processed(f, 0.0)

                last_heartbeat = time.time()

            if (time.time() - last_heartbeat) > timeout_threshold:
                print(f"[Watchdog] Stream {camera_id} HEARTBEAT LOST. Auto-Recovering...")
                break 
                
        video.release()
        time.sleep(1) 
        retry_count += 1

# Photogrammetry Controls
photogrammetryProc = None
generating = False
numPictures = 0

def run_photogrammetry():
    global generating, photogrammetryProc
    generating = True

    base_path = os.path.dirname(os.path.abspath(__file__))
    tool_path = os.path.join(base_path, "capture_tool")
    output_file = os.path.join(output_folder, "model.usdz")

    if not os.path.isfile(tool_path):
        print(f"[ERROR] Could not find compiled swift tool at: {tool_path}.")
        print("[Fix] Please run: swiftc -O -target arm64-apple-macos12 capture.swift -o capture_tool")
        generating = False
        return

    cmd = [tool_path, image_folder, output_file]

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
            print(line, end="")      
            logfile.write(line)      
        ret = proc.wait()

    photogrammetryProc = None
    if ret == 0:
        print(f"[System] PHOTOGRAMMETRY FINISHED. Saved to: {output_file}")
    else:
        print(f"[System] Photogrammetry exited with code {ret}")

    generating = False

# Threads
frames = []
threads = []

# Change depending on amount of cameras available
urls = [0, 0, 0, 0]

for i in range(len(urls)):
    frames.append(Frame())

    t = threading.Thread(target=run_camera, args=(urls[i], frames[i], i))
    t.daemon = True 
    threads.append(t)
    t.start()

    if i == 0:
        inf_thread = threading.Thread(target=run_inference, args=(model, frames[i]))
        inf_thread.daemon = True
        threads.append(inf_thread) 
        inf_thread.start()

log_thread = None
if ENABLE_LOGGING:
    log_thread = threading.Thread(target=telemetry_logger, args=(frames,))
    log_thread.daemon = True
    log_thread.start()

print("[System] All Vision Threads Active.")
print("[System] Controls: 'p' = Take Picture | 'g' = Start Photogrammetry | 'q' = Quit")

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
        
        # Capture keystroke (OpenCV window MUST be in focus)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'): 
            interrupt = True
            
        elif key == ord('p'):  # 'p' replaces the 'A' button
            print(f"[System] Image saved in images/img{numPictures}.png")
            filename = os.path.join(image_folder, f'img{numPictures}.jpg')

            # Jpeg for Meshroom
            success = cv2.imwrite(filename, imgs[0], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        
            if success:
                print(f"[System] Image successfully saved: {filename}")
            else:
                print(f"[ERROR] Failed to save image to: {filename}. Check folder permissions.")
            numPictures += 1
            
        elif key == ord('g'):  # 'g' replaces the 'B' button
            if generating:
                print('[System] Photogrammetry is already running')
            else:
                print('[System] Spinning up Photogrammetry Thread...')
                photogrammetryThread = threading.Thread(target=run_photogrammetry)
                photogrammetryThread.start()

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
    print("[System] Exit complete.\n")