#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DNN Face Extractor
File: dnn_face_extractor.py

描述:
这是一个工具脚本，用于从摄像头视频流中检测人脸，
并将检测到的人脸裁剪下来，保存为独立的图片文件。
这对于创建用于人脸识别训练的数据集非常有用。

!!! 重要 !!!
运行前，您必须下载两个DNN模型文件，并与此脚本放在同一个文件夹下:
1. 配置文件: deploy.prototxt.txt
2. 权重文件: res10_300x300_ssd_iter_140000.caffemodel
   (下载链接请参考 dnn_test_stream.py 脚本)

如何使用:
1.  确保已安装必要的库 (picamera2, opencv-python, numpy)。
2.  下载并放置好两个DNN模型文件。
3.  运行此脚本: `python3 dnn_face_extractor.py`
4.  程序会创建一个名为 `extracted_faces` 的文件夹。
5.  将您的脸对准摄像头，脚本会自动将检测到的面部图像保存到该文件夹中。
6.  按 `Ctrl+C` 停止程序。
"""

import cv2
import time
import os
import numpy as np

# --- 配置 ---
# DNN 模型文件路径
PROTOTXT_PATH = "deploy.prototxt.txt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
CONFIDENCE_THRESHOLD = 0.6  # 只有当置信度高于此值时才保存

# 摄像头配置
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# 输出配置
OUTPUT_DIR = "extracted_faces"
SAVE_INTERVAL = 0.5 # 每隔多少秒保存一次，避免瞬间保存太多重复照片
last_save_time = 0

# --- 主程序 ---
def main():
    global last_save_time

    # 检查模型文件
    if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(MODEL_PATH):
        print(f"[错误] 找不到DNN模型文件！请确保 '{PROTOTXT_PATH}' 和 '{MODEL_PATH}' 存在。")
        return

    # 创建输出文件夹
    if not os.path.exists(OUTPUT_DIR):
        print(f"[INFO] 正在创建输出文件夹: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)

    # 加载DNN模型
    print("[INFO] 正在加载DNN人脸检测模型...")
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    print("[INFO] 模型加载成功。")

    # 初始化摄像头
    picam2 = None
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
        picam2.configure(config)
        picam2.start()
        print("[INFO] Picamera2 摄像头初始化成功。")
    except Exception as e:
        print(f"[错误] Picamera2 初始化失败: {e}")
        print("请确保 picamera2 库已正确安装并且摄像头已连接。")
        return
    
    time.sleep(1.0)
    print("[INFO] 开始检测人脸... 按 Ctrl+C 退出。")

    try:
        while True:
            # 捕获帧
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            (h, w) = frame.shape[:2]

            # 创建 'blob' 并进行检测
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            # 循环遍历检测结果
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > CONFIDENCE_THRESHOLD:
                    # 控制保存频率
                    current_time = time.time()
                    if current_time - last_save_time < SAVE_INTERVAL:
                        continue
                    
                    last_save_time = current_time

                    # 计算边界框
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # 确保边界框不超出图像范围
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                    # 裁剪人脸
                    face_roi = frame[startY:endY, startX:endX]

                    if face_roi.size != 0:
                        # 保存图片
                        timestamp = int(time.time() * 1000)
                        filename = f"face_{timestamp}.jpg"
                        filepath = os.path.join(OUTPUT_DIR, filename)
                        cv2.imwrite(filepath, face_roi)
                        print(f"[成功] 已保存人脸截图: {filepath}")

    except KeyboardInterrupt:
        print("\n[INFO] 程序被用户中断。")
    finally:
        if picam2:
            picam2.stop()
        print("[INFO] 程序已关闭。")

if __name__ == '__main__':
    main()