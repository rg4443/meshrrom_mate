import cv2
from ultralytics import YOLO
import threading
import numpy as np
import pygame
import os
import shutil
import subprocess

image_folder = os.path.abspath("images")
output_folder = os.path.abspath("output")
pipeline = 'photogrammetryDraft'

# Remove old dirs (if present) then recreate empty
for p in (image_folder, output_folder):
    if os.path.exists(p):
        try:
            shutil.rmtree(p)
        except Exception as e:
            print(f"Failed to remove {p}: {e}")
    os.makedirs(p, exist_ok=True)

photogrammetryProc = None
def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    global photogrammetryProc
    try:
        photogrammetryProc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = photogrammetryProc.communicate()
        return photogrammetryProc.returncode, out.strip(), err.strip()
    except Exception as e:
        return -1, "", str(e)

class Frame:
    def __init__(self):
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.lock = threading.Lock()
    def set(self, frame):
        with self.lock:
            self.frame = frame.copy()
    def get(self):
        with self.lock:
            return self.frame.copy()

res = (1920, 1080)
res2 = (res[0]//3, res[1]//3)
def combine(imgs):
    global res, res2
    img1 = cv2.resize(imgs[0], res)
    img2 = cv2.resize(imgs[1], res2)
    img3 = cv2.resize(imgs[2], res2)
    # img4 = cv2.resize(imgs[3], res2)
    img4 = img3
    img5 = cv2.hconcat([img2, img3, img4])
    return cv2.vconcat([img1, img5])

interrupt = False
model = YOLO("best.pt")

def run_camera(url, frame, model=None):
    global interrupt
    try:
        video = cv2.VideoCapture(url)
        while not interrupt:
            r, f = video.read()
            if not r or f is None:
                continue
            if model is None:
                frame.set(f)
            else:
                results = model.predict(f)
                plotted = results[0].plot()

                # Count Crabs on Upper Left Corner
                number = len(results[0].boxes)
                cv2.putText(plotted, f"Green Crabs Detected: {number}", (7, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                frame.set(plotted)
        video.release()
    except Exception as e:
        print(f"ERROR: {e}")

controller = None
def connect_controller():
    global controller
    if pygame.joystick.get_count() > 0 and controller is None:
        controller = pygame.joystick.Joystick(0)
        controller.init()
        print(f'Controller connected: {controller.get_name()}')
    elif pygame.joystick.get_count() == 0 and controller is not None:
        print('Controller disconnected')
        controller = None

generating = False
numPictures = 0
def run_photogrammetry():
    global generating, numPictures, photogrammetryProc

    generating = True

    cmd = [
        "./meshroom_batch",
        "-i", image_folder,
        "-o", output_folder,
        "--pipeline", pipeline
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
        print("PHOTOGRAMMETRY FINISHED")
    elif ret != -9:
        print(f"Photogrammetry exited with code {ret}")

    generating = False


frames = []
threads = []
urls = [
    "udp://192.168.2.1:50000?fifo_size=1000000&overrun_nonfatal=1",
    "udp://192.168.2.1:50001?overrun_nonfatal=1", 
    0,
    # "udp://192.168.2.1:1986?overrun_nonfatal=1", 
    # "udp://192.168.2.1:1987?overrun_nonfatal=1",
]

for i in range(len(urls)):
    frames.append(Frame())
    if i == 0:
        threads.append(threading.Thread(target=run_camera, args=(urls[i], frames[i], model)))
    else:
        threads.append(threading.Thread(target=run_camera, args=(urls[i], frames[i])))
    threads[i].start()

pictureWasPressed = True
generateWasPressed = True
photogrammetryThread = None

pygame.init()
pygame.joystick.init()
connect_controller()
if controller is None:
    print("No controller connected\nPlug in a controller to use photogrammetry")
while not interrupt:
    try:
        imgs = [f.get() for f in frames]
        combined = combine(imgs)
        cv2.imshow('Frontend', combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            interrupt = True
        pygame.event.pump()
        connect_controller()
        if controller is not None:
            picture = bool(controller.get_button(0)) # A
            generate = bool(controller.get_button(1)) # B
            if picture and not pictureWasPressed:
                print(f"image saved in img{numPictures}")
                cv2.imwrite(f'images/img{numPictures}.png', imgs[0])

                print(f'Saved image {numPictures}')
                numPictures += 1
            if generate and not generateWasPressed:
                if generating:
                    print('Photogrammetry is already running')
                else:
                    photogrammetryThread = threading.Thread(target=run_photogrammetry)
                    photogrammetryThread.start()
            pictureWasPressed, generateWasPressed = picture, generate
            
    except KeyboardInterrupt:
        print("Got KeyboardInterrupt")

        interrupt = True
    except Exception as e:
        print(f'ERROR: {e}')
        interrupt = True

interrupt = True
for t in threads:
    t.join()
if photogrammetryThread is not None and photogrammetryThread.is_alive():
    if photogrammetryProc:
        photogrammetryProc.kill()
    photogrammetryThread.join()
cv2.destroyAllWindows()
pygame.quit()
print()
