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

CAMERA_INDEX        = 0        # 0 = default, 1/2/3 = other USB cameras
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

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera index {CAMERA_INDEX}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    # Keep buffer minimal so we always get the freshest frame
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # ── Shared state between display thread and OCR thread ────────────────
    lock           = threading.Lock()
    latest_frame   = [None]       # newest raw frame from camera
    overlay_frame  = [None]       # last annotated frame from OCR
    scan_num       = [0]
    ocr_running    = [False]
    stop_flag      = [False]

    # ── OCR worker — runs in background, never blocks display ─────────────
    def ocr_worker(frame_snapshot, snum):
        processed = preprocess(frame_snapshot)
        clean_lines, word_data, elapsed = run_ocr(processed)
        print_result(clean_lines, word_data, elapsed, snum)

        # Build annotated frame
        if HIGHLIGHT_WORDS:
            annotated = draw_highlights(frame_snapshot, word_data)
        else:
            annotated = frame_snapshot.copy()

        words_total = sum(len(l.split()) for l in clean_lines)
        info = (f"scan#{snum}  {elapsed:.2f}s  "
                f"conf>={CONFIDENCE_THRESHOLD}%  words:{words_total}")
        cv2.putText(annotated, info, (8, 22),
                    cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 255, 180), 1, cv2.LINE_AA)

        with lock:
            overlay_frame[0] = annotated
            ocr_running[0]   = False

    # ── Camera read thread — keeps latest_frame always fresh ──────────────
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
            now = time.time()

            # ── Grab latest frame (non-blocking) ─────────────────────────
            with lock:
                frame = latest_frame[0].copy() if latest_frame[0] is not None else None
                current_overlay = overlay_frame[0]
                is_scanning     = ocr_running[0]

            if frame is None:
                time.sleep(0.01)
                continue

            # ── Trigger OCR scan on interval (non-blocking) ───────────────
            if not is_scanning and (now - last_scan) >= SCAN_INTERVAL_SEC:
                last_scan = now
                with lock:
                    ocr_running[0] = True
                    scan_num[0]   += 1
                    snum = scan_num[0]
                t = threading.Thread(target=ocr_worker, args=(frame.copy(), snum), daemon=True)
                t.start()

            # ── Display ───────────────────────────────────────────────────
            if SHOW_PREVIEW_WINDOW:
                # Show overlay if available, otherwise raw feed
                if current_overlay is not None:
                    display = current_overlay.copy()
                    # Stamp LIVE indicator so user knows feed is fresh
                    cv2.circle(display, (display.shape[1] - 18, 14), 6, (0, 0, 255), -1)
                    cv2.putText(display, "LIVE", (display.shape[1] - 48, 19),
                                cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 200, 255), 1)
                else:
                    display = frame.copy()
                    cv2.putText(display, "Starting...", (8, 22),
                                cv2.FONT_HERSHEY_PLAIN, 1.1, (200, 200, 200), 1)

                # Scanning indicator
                if is_scanning:
                    cv2.putText(display, "SCANNING...", (8, display.shape[0] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 180), 1)

                cv2.imshow("OCR Live — press Q to quit", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
            else:
                # No display — just yield CPU
                time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        stop_flag[0] = True
        cap.release()
        if SHOW_PREVIEW_WINDOW:
            cv2.destroyAllWindows()
        print("\nStopped.")

if __name__ == "__main__":
    main()
