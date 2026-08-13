"""
截图工具 - 按 q 截图保存到数据集目录
"""
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import mss
from pynput import keyboard

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "datasets", "delta_force", "images", "train")
os.makedirs(SAVE_DIR, exist_ok=True)

# mss monitor index: 0=全部屏幕, 1=主显示器, 2=副显示器
MONITOR = 2

capture_counter = 0


def capture_screenshot():
    global capture_counter
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"delta_{timestamp}_{capture_counter:04d}.png"
    filepath = os.path.join(SAVE_DIR, filename)

    try:
        with mss.MSS() as sct:
            img = sct.grab(sct.monitors[MONITOR])
            frame = np.array(img)[:, :, :3]  # BGRA -> BGR
            cv2.imwrite(filepath, frame)
    except Exception as e:
        print(f"[!] 截图失败: {e}")
        return None

    capture_counter += 1
    print(f"[+] {filename} (累计{capture_counter}张)")
    return filepath


def on_press(key):
    try:
        if hasattr(key, 'char') and key.char == 'q':
            capture_screenshot()
    except Exception as e:
        print(f"按键错误: {e}")


def main():
    print("=" * 50)
    print("三角洲行动 截图工具")
    print("=" * 50)
    print(f"保存到: {SAVE_DIR}")
    print("\n操作说明:")
    print("  [q]    - 截图")
    print("  [Ctrl+C] - 退出\n")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    try:
        while listener.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
    print("\n退出截图工具")


if __name__ == "__main__":
    main()
