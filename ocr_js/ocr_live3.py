#!/usr/bin/env python3
"""
ocr_live.py — Live camera OCR, terminal output
Powered by Tesseract + OpenCV  (fully offline)

Install dependencies:
    pip install opencv-python pytesseract numpy
    sudo apt install tesseract-ocr
"""

# ─────────────────────────────────────────────────────────────────────────────
#  PARAMETERS — adjust these to tune detection
# ─────────────────────────────────────────────────────────────────────────────

CAMERA_INDEX        = 2      # 0 = default, 1/2/3 = other USB cameras
CAPTURE_WIDTH       = 1280     # camera resolution width
CAPTURE_HEIGHT      = 720      # camera resolution height

# OCR filtering
CONFIDENCE_THRESHOLD = 55      # 0–100: drop words below this Tesseract confidence
MIN_WORD_LENGTH      = 2       # drop words shorter than this many characters
NOISE_SYMBOLS_ONLY   = True    # drop words that are purely symbols/punctuation (|©=—_ etc.)
STRIP_SINGLE_DIGITS  = True    # drop lone single-digit tokens (common junk)

# Image pre-processing
GRAYSCALE            = True    # convert to grayscale before OCR
SHARPEN              = True    # unsharp-mask sharpening (helps dark/small labels)
SHARPEN_AMOUNT       = 1.5     # sharpening strength (1.0 = subtle, 2.5 = aggressive)
CONTRAST_STRETCH     = True    # auto contrast stretch (only if image range < 200)

# Scan behaviour
SCAN_INTERVAL_SEC    = 1.5     # seconds between automatic scans in live mode
SHOW_PREVIEW_WINDOW  = True    # show OpenCV window with camera + highlight boxes
HIGHLIGHT_WORDS      = True    # draw green/yellow boxes around detected words
PRINT_SEPARATOR      = True    # print a divider line between scans
CLEAR_TERMINAL       = False   # clear terminal before each scan output

# Tesseract config
TESS_LANG            = "eng"   # language code
TESS_PSM             = 3       # page segmentation: 3=auto, 6=single block, 11=sparse
#   PSM quick guide:
#     3  — fully automatic (best for unknown layouts)
#     6  — assume a single uniform block of text
#     11 — sparse text, find as much text as possible (good for labels/PCBs)
#     12 — sparse text with OSD

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
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
#  TKINTER DISPLAY  (replaces cv2.imshow — no Qt dependency, uses GTK on Ubuntu)
# ─────────────────────────────────────────────────────────────────────────────

_tk_root      = None
_tk_label     = None
_tk_ready     = threading.Event()
_tk_stop      = threading.Event()
_tk_queue     = [None]          # latest frame to display (BGR numpy array)
_tk_lock      = threading.Lock()

def _tk_thread_fn():
    """Runs the Tkinter event loop in its own thread."""
    global _tk_root, _tk_label
    try:
        import tkinter as tk
        from PIL import Image, ImageTk
    except ImportError:
        print("[display] ERROR: Pillow not installed. Run:  pip install Pillow")
        _tk_ready.set()
        return

    _tk_root = tk.Tk()
    _tk_root.title("OCR Live — press Q to quit")
    _tk_root.configure(bg='black')
    _tk_root.protocol("WM_DELETE_WINDOW", _tk_close)

    _tk_label = tk.Label(_tk_root, bg='black', cursor='none')
    _tk_label.pack()

    # Bind Q / Escape to quit
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
            # Convert BGR → RGB → PIL → ImageTk
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(img)
            _tk_label.configure(image=photo)
            _tk_label.image = photo   # keep reference
        _tk_root.after(16, _refresh)  # ~60 fps poll

    _tk_root.after(0, _refresh)
    _tk_root.mainloop()

def _tk_close():
    _tk_stop.set()

def start_display_window():
    """Start the Tkinter window in a background thread. Returns True if ok."""
    if not SHOW_PREVIEW_WINDOW:
        return False
    t = threading.Thread(target=_tk_thread_fn, daemon=True)
    t.start()
    _tk_ready.wait(timeout=5)
    if _tk_root is None:
        print("[display] Tkinter window failed to start — terminal-only mode.")
        return False
    print("[display] Preview window: OK (Tkinter/GTK)")
    return True

def display_frame(frame: np.ndarray):
    """Push a frame to the display window (non-blocking)."""
    with _tk_lock:
        _tk_queue[0] = frame

def display_alive() -> bool:
    return SHOW_PREVIEW_WINDOW and not _tk_stop.is_set()

# ─────────────────────────────────────────────────────────────────────────────
#  NOISE FILTER
# ─────────────────────────────────────────────────────────────────────────────

NOISE_RE = re.compile(r'^[|©=\-_+<>{}\[\]\\\/\^~`\'".,;:!?@#$%&*()\s]+$')

def is_noise(word: str) -> bool:
    if len(word) < MIN_WORD_LENGTH:
        return True
    if NOISE_SYMBOLS_ONLY and NOISE_RE.match(word):
        return True
    if STRIP_SINGLE_DIGITS and re.match(r'^\d$', word):
        return True
    return False

# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE PRE-PROCESSING  (mirrors the HTML JS pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(frame: np.ndarray) -> np.ndarray:
    img = frame.copy()

    if GRAYSCALE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if SHARPEN:
        # Unsharp mask: out = orig + amount * (orig - blur)
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.addWeighted(img, 1.0 + SHARPEN_AMOUNT, blurred, -SHARPEN_AMOUNT, 0)
        img = np.clip(img, 0, 255).astype(np.uint8)

    if CONTRAST_STRETCH:
        vmin, vmax = int(img.min()), int(img.max())
        if (vmax - vmin) < 200:          # only stretch low-contrast images
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    return img

# ─────────────────────────────────────────────────────────────────────────────
#  OCR
# ─────────────────────────────────────────────────────────────────────────────

TESS_CONFIG = f"--psm {TESS_PSM} --oem 1 -l {TESS_LANG}"

def run_ocr(processed: np.ndarray):
    """Run Tesseract, return (clean_lines, word_data, elapsed)."""
    t0 = time.time()
    data = pytesseract.image_to_data(
        processed,
        config=TESS_CONFIG,
        output_type=pytesseract.Output.DICT
    )
    elapsed = time.time() - t0

    # Group words into lines, filter noise
    # Tesseract data: each entry has block_num, par_num, line_num, word_num, text, conf
    lines_dict: dict[tuple, list] = {}
    word_data = []  # for highlight drawing

    n = len(data['text'])
    for i in range(n):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        if conf < 0:          # -1 = non-word (layout element)
            continue
        if conf < CONFIDENCE_THRESHOLD:
            continue
        if is_noise(text):
            continue

        line_key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
        lines_dict.setdefault(line_key, []).append(text)

        # bbox for highlight
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        word_data.append({
            'text': text,
            'conf': conf,
            'bbox': (x, y, x + w, y + h)
        })

    clean_lines = [' '.join(words) for words in lines_dict.values() if words]
    return clean_lines, word_data, elapsed

# ─────────────────────────────────────────────────────────────────────────────
#  DRAW HIGHLIGHTS ON ORIGINAL FRAME
# ─────────────────────────────────────────────────────────────────────────────

def draw_highlights(frame: np.ndarray, word_data: list) -> np.ndarray:
    out = frame.copy()
    for w in word_data:
        x0, y0, x1, y1 = w['bbox']
        color = (0, 255, 120) if w['conf'] > 75 else (0, 220, 255)  # green / yellow
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 1)
        cv2.putText(out, f"{w['conf']}%", (x0, max(0, y0 - 3)),
                    cv2.FONT_HERSHEY_PLAIN, 0.7, color, 1, cv2.LINE_AA)
    return out

# ─────────────────────────────────────────────────────────────────────────────
#  TERMINAL PRINT
# ─────────────────────────────────────────────────────────────────────────────

def print_result(clean_lines: list, word_data: list, elapsed: float, scan_num: int):
    if CLEAR_TERMINAL:
        os.system('clear')

    ts = datetime.now().strftime('%H:%M:%S')
    words_found = sum(len(l.split()) for l in clean_lines)
    chars_found = sum(len(l.replace(' ', '')) for l in clean_lines)

    if PRINT_SEPARATOR:
        print(f"\n{'─'*60}")

    print(f"[{ts}]  scan #{scan_num}  |  {elapsed:.2f}s  |  "
          f"{words_found} words  {chars_found} chars  {len(clean_lines)} lines")

    if clean_lines:
        print()
        for line in clean_lines:
            print(f"  {line}")
    else:
        print("  (no text detected — try lowering CONFIDENCE_THRESHOLD or MIN_WORD_LENGTH)")

    sys.stdout.flush()

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"OCR LIVE — camera {CAMERA_INDEX}  |  conf≥{CONFIDENCE_THRESHOLD}%  "
          f"|  minLen≥{MIN_WORD_LENGTH}  |  PSM {TESS_PSM}")
    print(f"Sharpen={'ON' if SHARPEN else 'OFF'}  |  "
          f"Grayscale={'ON' if GRAYSCALE else 'OFF'}  |  "
          f"Interval={SCAN_INTERVAL_SEC}s")
    print("Press  Q  in preview window (or Ctrl+C) to quit.\n")

    # Start display window (Tkinter, no Qt)
    use_display = start_display_window()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera index {CAMERA_INDEX}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # ── Shared state ──────────────────────────────────────────────────────
    lock          = threading.Lock()
    latest_frame  = [None]
    overlay_frame = [None]
    scan_num      = [0]
    ocr_running   = [False]
    stop_flag     = [False]

    # ── OCR worker ────────────────────────────────────────────────────────
    def ocr_worker(frame_snapshot, snum):
        processed = preprocess(frame_snapshot)
        clean_lines, word_data, elapsed = run_ocr(processed)
        print_result(clean_lines, word_data, elapsed, snum)

        annotated = draw_highlights(frame_snapshot, word_data) if HIGHLIGHT_WORDS \
                    else frame_snapshot.copy()

        words_total = sum(len(l.split()) for l in clean_lines)
        info = (f"scan#{snum}  {elapsed:.2f}s  "
                f"conf>={CONFIDENCE_THRESHOLD}%  words:{words_total}")
        cv2.putText(annotated, info, (8, 22),
                    cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 255, 180), 1, cv2.LINE_AA)

        with lock:
            overlay_frame[0] = annotated
            ocr_running[0]   = False

    # ── Camera reader thread ──────────────────────────────────────────────
    def camera_reader():
        while not stop_flag[0]:
            ret, frame = cap.read()
            if ret:
                with lock:
                    latest_frame[0] = frame
            else:
                time.sleep(0.01)

    reader_thread = threading.Thread(target=camera_reader, daemon=True)
    reader_thread.start()

    # Wait for first frame
    print("Waiting for camera...")
    for _ in range(50):
        with lock:
            if latest_frame[0] is not None:
                break
        time.sleep(0.1)
    else:
        print("ERROR: No frames received from camera")
        stop_flag[0] = True
        cap.release()
        sys.exit(1)
    print("Camera ready.\n")

    last_scan = 0.0

    try:
        while True:
            # Exit if display window was closed
            if use_display and not display_alive():
                print("Window closed.")
                break

            now = time.time()

            with lock:
                frame           = latest_frame[0].copy() if latest_frame[0] is not None else None
                current_overlay = overlay_frame[0]
                is_scanning     = ocr_running[0]

            if frame is None:
                time.sleep(0.01)
                continue

            # ── Trigger OCR ───────────────────────────────────────────────
            if not is_scanning and (now - last_scan) >= SCAN_INTERVAL_SEC:
                last_scan = now
                with lock:
                    ocr_running[0] = True
                    scan_num[0]   += 1
                    snum = scan_num[0]
                t = threading.Thread(target=ocr_worker,
                                     args=(frame.copy(), snum), daemon=True)
                t.start()

            # ── Push frame to display ─────────────────────────────────────
            if use_display:
                if current_overlay is not None:
                    disp = current_overlay.copy()
                    # LIVE dot (top-right)
                    h, w = disp.shape[:2]
                    cv2.circle(disp, (w - 18, 14), 6, (0, 0, 255), -1)
                    cv2.putText(disp, "LIVE", (w - 50, 19),
                                cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 200, 255), 1)
                else:
                    disp = frame.copy()
                    cv2.putText(disp, "Starting...", (8, 22),
                                cv2.FONT_HERSHEY_PLAIN, 1.1, (200, 200, 200), 1)

                if is_scanning:
                    h = disp.shape[0]
                    cv2.putText(disp, "SCANNING...", (8, h - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 180), 1)

                display_frame(disp)

            time.sleep(0.01)   # ~100 fps cap, keeps CPU sane

    except KeyboardInterrupt:
        pass
    finally:
        stop_flag[0] = True
        _tk_stop.set()
        cap.release()
        print("\nStopped.")

if __name__ == "__main__":
    main()
