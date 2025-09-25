#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DNN Face Detection Live Stream Test
File: dnn_test_stream.py

描述:
这个脚本使用一个基于深度神经网络 (DNN) 的模型来检测人脸，
它比 Haar Cascade 更强大，能够检测不同角度和光照下的人脸。
同样，它会创建一个Web服务器来流式传输处理后的视频。

!!! 重要 !!!
运行前，您必须下载两个模型文件，并与此脚本放在同一个文件夹下:
1. 配置文件: deploy.prototxt.txt
   下载链接: https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt.txt
2. 权重文件: res10_300x300_ssd_iter_140000.caffemodel
   下载链接: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

如何使用:
1.  确保您已经安装了必要的库 (picamera2, opencv-python)。
2.  下载上面列出的两个模型文件。
3.  在树莓派终端中运行此脚本:
    python3 dnn_test_stream.py
4.  在电脑浏览器中打开脚本打印出的链接。
"""

import cv2
import http.server
import socketserver
import threading
import time
import os
import socket
import numpy as np

# --- 全局配置 ---
HOST = ''
PORT = 8000
# --- DNN 模型文件路径 ---
PROTOTXT_PATH = "deploy.prototxt.txt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
CONFIDENCE_THRESHOLD = 0.5  # 置信度阈值，可以调整

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 70

# --- 全局变量 ---
output_frame = None
lock = threading.Lock()
stop_threads = False

# --- 视觉处理线程 (已升级为DNN模型) ---
def video_processing_thread():
    global output_frame, lock, stop_threads

    # 检查模型文件是否存在
    if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(MODEL_PATH):
        print("[错误] 找不到DNN模型文件！")
        print(f"请确保 '{PROTOTXT_PATH}' 和 '{MODEL_PATH}' 文件与脚本在同一目录下。")
        print("您可以从脚本顶部的注释中找到下载链接。")
        return

    # 加载DNN模型
    print("[INFO] 正在加载DNN人脸检测模型...")
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    print("[INFO] 模型加载成功。")

    # --- 初始化摄像头 (与之前相同) ---
    cap = None
    picam2 = None
    try:
        from picamera2 import Picamera2
        print("[INFO] 检测到 Picamera2 库，尝试使用 libcamera...")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
        picam2.configure(config)
        picam2.start()
        print("[INFO] Picamera2 摄像头初始化成功。")
    except (ImportError, RuntimeError) as e:
        print(f"[警告] Picamera2 初始化失败: {e}. 回退到 cv2.VideoCapture。")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[错误] 使用 cv2.VideoCapture 也无法打开摄像头。")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    time.sleep(1)
    print("[INFO] 视觉处理线程已启动，正在捕获视频...")

    while not stop_threads:
        if picam2:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif cap:
            ret, frame = cap.read()
            if not ret:
                print("[警告] 无法读取视频帧。")
                time.sleep(0.5)
                continue
        else:
            break

        (h, w) = frame.shape[:2]
        
        # --- DNN处理流程 ---
        # 1. 创建一个 'blob' (图像预处理)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # 2. 将 blob 输入网络并获取检测结果
        net.setInput(blob)
        detections = net.forward()

        # 3. 循环遍历检测到的人脸
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # 4. 过滤掉置信度低的检测
            if confidence > CONFIDENCE_THRESHOLD:
                # 5. 计算边界框坐标并绘制
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                text = f"{confidence * 100:.2f}%"
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # 编码并更新输出帧
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        with lock:
            output_frame = buffer.tobytes()

    if picam2: picam2.stop()
    if cap: cap.release()
    print("[INFO] 视觉处理线程已停止。")


# --- Web 服务器处理程序 (无变化) ---
class StreamingHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        global output_frame, lock
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html_content = b"""
            <html><head><title>DNN Test Stream</title></head>
            <body style="margin:0; background:#333;"><img src="/stream.mjpeg" width="100%" /></body>
            </html>"""
            self.wfile.write(html_content)
        elif self.path == '/stream.mjpeg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with lock:
                        frame = output_frame
                    if frame:
                        self.wfile.write(b'--FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(frame))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                    time.sleep(0.03)
            except Exception as e:
                print(f"Removed client: {self.client_address}, reason: {e}")
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

def get_ip_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# --- 主程序入口 (无变化) ---
if __name__ == '__main__':
    process_thread = threading.Thread(target=video_processing_thread, daemon=True)
    process_thread.start()
    server = None
    try:
        address = (HOST, PORT)
        server = StreamingServer(address, StreamingHandler)
        ip_address = get_ip_address()
        print("[INFO] 服务器已启动!")
        print(f"请在您的电脑浏览器中打开: http://{ip_address}:{PORT}")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] 检测到 Ctrl+C，正在关闭程序...")
    except Exception as e:
        print(f"[错误] 启动服务器失败: {e}")
    finally:
        stop_threads = True
        if server:
            server.shutdown()
            server.server_close()
        process_thread.join(timeout=2)
        print("[INFO] 程序已安全关闭。")
