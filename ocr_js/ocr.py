#!/usr/bin/env python3
"""
ocr_live_fast.py — Optimized Live Camera OCR (Low-Lag Version)
Powered by Tesseract + OpenCV

Install:
    pip install opencv-python pytesseract numpy Pillow
    sudo apt install tesseract-ocr
"""

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

CAMERA_INDEX = 2

# LOWER RESOLUTION = MUCH FASTER
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480

# OCR FILTERING
CONFIDENCE_THRESHOLD = 55
MIN_WORD_LENGTH = 2
NOISE_SYMBOLS_ONLY = True
STRIP_SINGLE_DIGITS = True

# IMAGE PROCESSING
GRAYSCALE = True

# DISABLED FOR SPEED
SHARPEN = False
SHARPEN_AMOUNT = 1.5

CONTRAST_STRETCH = False

# SCAN SETTINGS
SCAN_INTERVAL_SEC = 2.5

SHOW_PREVIEW_WINDOW = True
HIGHLIGHT_WORDS = True

PRINT_SEPARATOR = True
CLEAR_TERMINAL = False

# FASTER PSM
TESS_LANG = "eng"
TESS_PSM = 11

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
import pytesseract
import time
import os
import re
import sys
import threading
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# TKINTER DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

_tk_root = None
_tk_label = None
_tk_ready = threading.Event()
_tk_stop = threading.Event()

_tk_queue = [None]
_tk_lock = threading.Lock()

def _tk_thread_fn():
    global _tk_root, _tk_label

    try:
        import tkinter as tk
        from PIL import Image, ImageTk
    except ImportError:
        print("[display] ERROR: pip install Pillow")
        _tk_ready.set()
        return

    _tk_root = tk.Tk()
    _tk_root.title("OCR LIVE FAST")
    _tk_root.configure(bg='black')

    _tk_root.protocol("WM_DELETE_WINDOW", _tk_close)

    _tk_label = tk.Label(_tk_root, bg='black')
    _tk_label.pack()

    _tk_root.bind('<q>', lambda e: _tk_close())
    _tk_root.bind('<Escape>', lambda e: _tk_close())

    _tk_ready.set()

    def _refresh():

        if _tk_stop.is_set():
            _tk_root.destroy()
            return

        with _tk_lock:
            frame = _tk_queue[0]

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(img)

            _tk_label.configure(image=photo)
            _tk_label.image = photo

        # Reduced refresh load
        _tk_root.after(30, _refresh)

    _tk_root.after(0, _refresh)
    _tk_root.mainloop()

def _tk_close():
    _tk_stop.set()

def start_display_window():

    if not SHOW_PREVIEW_WINDOW:
        return False

    t = threading.Thread(target=_tk_thread_fn, daemon=True)
    t.start()

    _tk_ready.wait(timeout=5)

    if _tk_root is None:
        print("[display] Failed to start preview")
        return False

    print("[display] Preview window started")
    return True

def display_frame(frame):

    with _tk_lock:
        _tk_queue[0] = frame

def display_alive():
    return SHOW_PREVIEW_WINDOW and not _tk_stop.is_set()

# ─────────────────────────────────────────────────────────────────────────────
# NOISE FILTER
# ─────────────────────────────────────────────────────────────────────────────

NOISE_RE = re.compile(
    r'^[|©=\-_+<>{}\[\]\\\/\^~`\'".,;:!?@#$%&*()\s]+$'
)

def is_noise(word):

    if len(word) < MIN_WORD_LENGTH:
        return True

    if NOISE_SYMBOLS_ONLY and NOISE_RE.match(word):
        return True

    if STRIP_SINGLE_DIGITS and re.match(r'^\d$', word):
        return True

    return False

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(frame):

    img = frame.copy()

    if GRAYSCALE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if SHARPEN:
        blurred = cv2.GaussianBlur(img, (3, 3), 0)

        img = cv2.addWeighted(
            img,
            1.0 + SHARPEN_AMOUNT,
            blurred,
            -SHARPEN_AMOUNT,
            0
        )

        img = np.clip(img, 0, 255).astype(np.uint8)

    if CONTRAST_STRETCH:
        vmin, vmax = int(img.min()), int(img.max())

        if (vmax - vmin) < 200:
            img = cv2.normalize(
                img,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            )

    return img

# ─────────────────────────────────────────────────────────────────────────────
# OCR
# ─────────────────────────────────────────────────────────────────────────────

TESS_CONFIG = f"--psm {TESS_PSM} --oem 1 -l {TESS_LANG}"

def run_ocr(processed):

    t0 = time.time()

    data = pytesseract.image_to_data(
        processed,
        config=TESS_CONFIG,
        output_type=pytesseract.Output.DICT
    )

    elapsed = time.time() - t0

    lines_dict = {}
    word_data = []

    n = len(data['text'])

    for i in range(n):

        text = data['text'][i].strip()

        try:
            conf = int(float(data['conf'][i]))
        except:
            continue

        if conf < 0:
            continue

        if conf < CONFIDENCE_THRESHOLD:
            continue

        if is_noise(text):
            continue

        line_key = (
            data['block_num'][i],
            data['par_num'][i],
            data['line_num'][i]
        )

        lines_dict.setdefault(line_key, []).append(text)

        x = data['left'][i]
        y = data['top'][i]
        w = data['width'][i]
        h = data['height'][i]

        word_data.append({
            'text': text,
            'conf': conf,
            'bbox': (x, y, x + w, y + h)
        })

    clean_lines = [
        ' '.join(words)
        for words in lines_dict.values()
        if words
    ]

    return clean_lines, word_data, elapsed

# ─────────────────────────────────────────────────────────────────────────────
# DRAW BOXES
# ─────────────────────────────────────────────────────────────────────────────

def draw_highlights(frame, word_data):

    out = frame.copy()

    for w in word_data:

        x0, y0, x1, y1 = w['bbox']

        color = (
            (0, 255, 120)
            if w['conf'] > 75
            else (0, 220, 255)
        )

        cv2.rectangle(out, (x0, y0), (x1, y1), color, 1)

    return out

# ─────────────────────────────────────────────────────────────────────────────
# PRINT RESULT
# ─────────────────────────────────────────────────────────────────────────────

def print_result(clean_lines, elapsed, scan_num):

    if CLEAR_TERMINAL:
        os.system('clear')

    ts = datetime.now().strftime('%H:%M:%S')

    words_found = sum(len(l.split()) for l in clean_lines)

    chars_found = sum(len(l.replace(' ', '')) for l in clean_lines)

    if PRINT_SEPARATOR:
        print(f"\n{'─'*60}")

    print(
        f"[{ts}] scan #{scan_num} | "
        f"{elapsed:.2f}s | "
        f"{words_found} words | "
        f"{chars_found} chars"
    )

    print()

    if clean_lines:
        for line in clean_lines:
            print(f"  {line}")
    else:
        print("  No text detected")

    sys.stdout.flush()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():

    print("\nOCR LIVE FAST")
    print(f"Resolution: {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}")
    print(f"PSM: {TESS_PSM}")
    print(f"Scan interval: {SCAN_INTERVAL_SEC}s\n")

    use_display = start_display_window()

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    # VERY IMPORTANT
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    lock = threading.Lock()

    latest_frame = [None]

    overlay_frame = [None]

    scan_num = [0]

    ocr_running = [False]

    stop_flag = [False]

    # ─────────────────────────────────────────────────────
    # OCR THREAD
    # ─────────────────────────────────────────────────────

    def ocr_worker(frame_snapshot, snum):

        processed = preprocess(frame_snapshot)

        clean_lines, word_data, elapsed = run_ocr(processed)

        print_result(clean_lines, elapsed, snum)

        annotated = draw_highlights(frame_snapshot, word_data)

        with lock:
            overlay_frame[0] = annotated
            ocr_running[0] = False

    # ─────────────────────────────────────────────────────
    # CAMERA THREAD
    # ─────────────────────────────────────────────────────

    def camera_reader():

        while not stop_flag[0]:

            ret, frame = cap.read()

            if ret:
                with lock:
                    latest_frame[0] = frame

            else:
                time.sleep(0.01)

    threading.Thread(
        target=camera_reader,
        daemon=True
    ).start()

    print("Waiting for camera...")

    while latest_frame[0] is None:
        time.sleep(0.1)

    print("Camera ready.\n")

    last_scan = 0

    try:

        while True:

            if use_display and not display_alive():
                break

            now = time.time()

            with lock:

                frame = (
                    latest_frame[0].copy()
                    if latest_frame[0] is not None
                    else None
                )

                current_overlay = overlay_frame[0]

                is_scanning = ocr_running[0]

            if frame is None:
                continue

            # ─────────────────────────────────────────────
            # START OCR
            # ─────────────────────────────────────────────

            if (
                not is_scanning and
                (now - last_scan) >= SCAN_INTERVAL_SEC
            ):

                last_scan = now

                with lock:
                    ocr_running[0] = True

                    scan_num[0] += 1

                    snum = scan_num[0]

                threading.Thread(
                    target=ocr_worker,
                    args=(frame.copy(), snum),
                    daemon=True
                ).start()

            # ─────────────────────────────────────────────
            # LIVE DISPLAY (FIXED LAG VERSION)
            # ─────────────────────────────────────────────

            if use_display:

                # ALWAYS SHOW LIVE FRAME
                disp = frame.copy()

                # Blend OCR overlay slightly
                if current_overlay is not None:

                    alpha = 0.35

                    disp = cv2.addWeighted(
                        current_overlay,
                        alpha,
                        disp,
                        1 - alpha,
                        0
                    )

                h, w = disp.shape[:2]

                # LIVE DOT
                cv2.circle(
                    disp,
                    (w - 18, 14),
                    6,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    disp,
                    "LIVE",
                    (w - 50, 19),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.9,
                    (0, 200, 255),
                    1
                )

                if is_scanning:

                    cv2.putText(
                        disp,
                        "SCANNING...",
                        (8, h - 10),
                        cv2.FONT_HERSHEY_PLAIN,
                        1.0,
                        (0, 255, 180),
                        1
                    )

                display_frame(disp)

            time.sleep(0.005)

    except KeyboardInterrupt:
        pass

    finally:

        stop_flag[0] = True

        _tk_stop.set()

        cap.release()

        print("\nStopped.")

if __name__ == "__main__":
    main()
