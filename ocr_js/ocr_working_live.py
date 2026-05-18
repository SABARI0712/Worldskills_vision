#!/usr/bin/env python3
"""
FAST LIVE OCR
────────────────────────────────────────────────────────────

Features:
• Smooth realtime camera feed
• OCR output only in terminal
• Very low lag
• Lightweight CPU usage
• Fully offline
• Press Q to quit

Install:
    pip install opencv-python pytesseract

Ubuntu:
    sudo apt install tesseract-ocr
"""

# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────

import cv2
import pytesseract
import threading
import time

# ─────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────

CAMERA_INDEX = 2

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# OCR scan interval (seconds)
SCAN_INTERVAL = 2.5

# Tesseract settings
TESS_CONFIG = "--psm 11"

# ─────────────────────────────────────────────────────────────
# SHARED VARIABLES
# ─────────────────────────────────────────────────────────────

latest_frame = None

frame_lock = threading.Lock()

ocr_running = False

stop_flag = False

scan_counter = 0

# ─────────────────────────────────────────────────────────────
# OCR FUNCTION
# ─────────────────────────────────────────────────────────────

def run_ocr(frame, scan_id):

    global ocr_running

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # OCR start time
    t0 = time.time()

    # Run Tesseract
    text = pytesseract.image_to_string(
        gray,
        config=TESS_CONFIG
    )

    # OCR elapsed time
    elapsed = time.time() - t0

    # Clean output
    text = text.strip()

    # Terminal output
    print("\n" + "─" * 60)

    print(
        f"SCAN #{scan_id} "
        f"| OCR Time: {elapsed:.2f}s"
    )

    print()

    if text:
        print(text)
    else:
        print("No text detected")

    print("─" * 60)

    # OCR complete
    ocr_running = False

# ─────────────────────────────────────────────────────────────
# CAMERA READER THREAD
# ─────────────────────────────────────────────────────────────

def camera_reader(cap):

    global latest_frame
    global stop_flag

    while not stop_flag:

        ret, frame = cap.read()

        if ret:

            with frame_lock:
                latest_frame = frame

        else:
            time.sleep(0.01)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():

    global latest_frame
    global ocr_running
    global stop_flag
    global scan_counter

    print("\nFAST LIVE OCR")
    print("Press Q to quit\n")

    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    # Camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # IMPORTANT:
    # Prevents delayed buffered frames
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Start camera thread
    threading.Thread(
        target=camera_reader,
        args=(cap,),
        daemon=True
    ).start()

    # Wait for first frame
    print("Starting camera...")

    while latest_frame is None:
        time.sleep(0.1)

    print("Camera ready\n")

    last_scan_time = 0

    # ─────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────

    while True:

        # Get latest frame
        with frame_lock:

            if latest_frame is None:
                continue

            frame = latest_frame.copy()

        # ─────────────────────────────────────
        # LIVE CAMERA DISPLAY
        # ─────────────────────────────────────

        # LIVE text
        cv2.putText(
            frame,
            "LIVE",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Show camera
        cv2.imshow(
            "FAST LIVE OCR",
            frame
        )

        # ─────────────────────────────────────
        # OCR TIMER
        # ─────────────────────────────────────

        current_time = time.time()

        if (
            not ocr_running and
            (current_time - last_scan_time) >= SCAN_INTERVAL
        ):

            last_scan_time = current_time

            ocr_running = True

            scan_counter += 1

            # Start OCR thread
            threading.Thread(
                target=run_ocr,
                args=(frame.copy(), scan_counter),
                daemon=True
            ).start()

        # ─────────────────────────────────────
        # EXIT KEY
        # ─────────────────────────────────────

        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    # ─────────────────────────────────────────
    # CLEANUP
    # ─────────────────────────────────────────

    stop_flag = True

    cap.release()

    cv2.destroyAllWindows()

    print("\nStopped")

# ─────────────────────────────────────────────────────────────
# START
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
