#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Animal Robot Main Controller (V18 - Final Path Fix Version)
File: animal_robot.py

描述:
这是项目的最终修正版 V2。
1.  修正了自动将项目根目录添加到Python路径的逻辑，以解决 'NB3.Sound' 导入错误。
2.  已将您的 Google Cloud Project ID 和 Region 正确配置。
"""

import sys
import os
import serial
import time
import threading
import random
import json
import cv2
import numpy as np

# --- [代码修正] ---
# 将项目根目录 (LastBlackBox) 添加到Python的搜索路径
# 这能确保程序总能找到专有的 NB3 库
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 现在可以安全地导入其他模块了 ---
from vertex_ai_adapter import Vertex_AI_Adapter
from nb3_sound_adapter import NB3_Sound_Adapter

# --- 全局配置 ---
GCP_PROJECT_ID = "vernal-tracer-473212-h6"
GCP_REGION = "europe-west2"

SER_PORT = '/dev/ttyUSB0'
SER_BAUD = 115200
BRAIN_TICK_RATE = 0.1
TILT_COMMAND_COOLDOWN = 0.2
MOVE_COMMAND_COOLDOWN = 0.1
PROTOTXT_PATH = "deploy.prototxt.txt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
DNN_CONFIDENCE_THRESHOLD = 0.6
TRAINER_FILE = 'trainer.yml'
NAMES_FILE = 'names.json'
RECOGNITION_CONFIDENCE_THRESHOLD = 75
HUNTER_FACE_ID = "hunter_John"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
VERTICAL_TARGET_ZONE = (0.40, 0.60)
HORIZONTAL_TARGET_ZONE = (0.45, 0.55)
TARGET_FACE_HEIGHT_RATIO = 0.45
INTERACTION_TIMEOUT = 10.0
LANGUAGE_MODEL_FILE = 'language_model_en.json'

# ===================================================================
#  模块1: 视觉处理器 (FaceProcessor) - 无变化
# ===================================================================
class FaceProcessor(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.any_person_detected = False
        self.threat_detected = False
        self.main_face_bbox = None
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        
        # 这一行需要 opencv-contrib-python
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.names = {}
        if os.path.exists(TRAINER_FILE) and os.path.exists(NAMES_FILE):
            self.recognizer.read(TRAINER_FILE)
            with open(NAMES_FILE, 'r') as f: self.names = json.load(f)
        else: self.recognizer = None

        if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(MODEL_PATH):
             raise FileNotFoundError("[视觉][错误] 找不到DNN模型文件！")
        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

    def stop(self): self._stop_event.set()

    def run(self):
        picam2 = None
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
            picam2.configure(config)
            picam2.start()
            time.sleep(1.0)

            while not self._stop_event.is_set():
                frame = picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                (h, w) = frame.shape[:2]

                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                self.net.setInput(blob)
                detections = self.net.forward()
                
                detected_face_this_frame = False
                recognized_hunter_this_frame = False
                best_detection_info = None
                max_confidence = -1

                for i in range(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > DNN_CONFIDENCE_THRESHOLD:
                        detected_face_this_frame = True
                        if confidence > max_confidence:
                            max_confidence = confidence
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            best_detection_info = (startX, startY, endX-startX, endY-startY)
                
                current_face_bbox = None
                if best_detection_info:
                    current_face_bbox = best_detection_info
                    if self.recognizer:
                        (x, y, bw, bh) = best_detection_info
                        face_roi_gray = gray[y:y+bh, x:x+bw]
                        if face_roi_gray.size != 0:
                            id_, conf = self.recognizer.predict(face_roi_gray)
                            if conf < RECOGNITION_CONFIDENCE_THRESHOLD:
                                recognized_name = self.names.get(str(id_), "unknown")
                                if recognized_name == HUNTER_FACE_ID:
                                    recognized_hunter_this_frame = True
                
                with self.lock:
                    self.any_person_detected = detected_face_this_frame
                    self.threat_detected = recognized_hunter_this_frame
                    self.main_face_bbox = current_face_bbox
        except Exception as e:
            print(f"[视觉][严重错误] 视觉处理线程因异常而终止: {e}")
        finally:
            if picam2: picam2.stop()


# ===================================================================
#  模块2: Arduino 通信器 - 无变化
# ===================================================================
class ArduinoCommunicator:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.feed_button_pressed = False
        self._stop_event = threading.Event()
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)
            self.listener_thread = threading.Thread(target=self._listen_for_data, daemon=True)
            self.listener_thread.start()
            print(f"[通信] 成功连接到Arduino: {port}")
        except serial.SerialException as e:
            print(f"[通信][警告] 无法连接到Arduino {port}. {e}. 将以模拟模式运行。")

    def _listen_for_data(self):
        while not self._stop_event.is_set() and self.ser:
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8').strip()
                    if line == 'k':
                        print("[通信] 接收到喂食按钮信号！")
                        self.feed_button_pressed = True
            except (serial.SerialException, UnicodeDecodeError):
                time.sleep(1)

    def send_command(self, command):
        if self.ser: self.ser.write((command + '\n').encode('utf-8'))
    
    def read_and_reset_button_press(self):
        if self.feed_button_pressed:
            self.feed_button_pressed = False
            return True
        return False

    def close(self):
        self._stop_event.set()
        if self.ser and self.ser.is_open: self.ser.close()


# ===================================================================
#  模块3: 语言模型 - 无变化
# ===================================================================
class LanguageModel:
    def __init__(self, filepath):
        self.filepath = filepath
        self.sentences = {}
        self.load()

    def get_default_sentences(self):
        return {
            "hunt": [{"text": "I'm hungry, could you please press the button on my back to feed me?", "score": 10}],
            "escape": [{"text": "Please don't hurt me! Help!", "score": 10}],
            "reproduce": [{"text": "Thank you so much! Can I ask for one more little favor?", "score": 10}],
            "success_food": [{"text": "Yummy! Thank you!", "score": 10}],
            "success_reproduce": [{"text": "You are the best! Thank you!", "score": 10}],
            "failure": [{"text": "Oh, okay. Sorry to bother you.", "score": 10}]
        }

    def load(self):
        defaults = self.get_default_sentences()
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    loaded_sentences = json.load(f)
                self.sentences = {**defaults, **loaded_sentences}
                if len(self.sentences) > len(loaded_sentences):
                    self.save()
            except (json.JSONDecodeError, TypeError):
                self.sentences = defaults
                self.save()
        else:
            self.sentences = defaults
            self.save()

    def save(self):
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.sentences, f, ensure_ascii=False, indent=4)
            
    def get_sentence(self, category):
        options = self.sentences.get(category, [])
        if not options: 
            print(f"[语言][警告] 在模型中找不到类别 '{category}'")
            return "..."
        total_score = sum(item['score'] for item in options)
        if total_score == 0: return random.choice(options)['text']
        pick = random.uniform(0, total_score)
        current = 0
        for item in options:
            current += item['score']
            if current > pick: return item['text']
        return options[-1]['text']
        
    def update_sentence_score(self, category, sentence_text, success):
        for item in self.sentences.get(category, []):
            if item['text'] == sentence_text:
                if success: item['score'] = max(1, item['score'] + 5)
                else: item['score'] = max(1, item['score'] - 3)
                break
        self.save()

# ===================================================================
#  模块4: 服务 - 无变化
# ===================================================================
class Services:
    def __init__(self):
        self.sound_adapter = NB3_Sound_Adapter()
        print("[服务] 服务模块已初始化，使用NB3声音适配器。")

    def speak(self, text):
        print(f"** Robot says: '{text}' **")
        if text and text != "...":
            self.sound_adapter.speak(text)
        else:
            print("[服务][警告] 检测到空的文本，已阻止播放。")

    def play_sound(self, sound_type):
        print(f"vvv Playing '{sound_type}' sound vvv")
        self.sound_adapter.play_sound(sound_type)

    def listen_for_speech(self, timeout=5):
        return self.sound_adapter.listen(timeout=timeout)
    
    def shutdown(self):
        self.sound_adapter.shutdown()

# ===================================================================
#  模块5: 机器人大脑 (AnimalRobot) - 已更新为使用Vertex AI
# ===================================================================
class AnimalRobot:
    def __init__(self):
        self.state = "HUNTING_SEARCHING"
        self.interaction_start_time = 0
        self.current_sentence = ""
        self.last_tilt_command_time = 0
        self.last_move_command_time = 0

        print("--- Animal Robot V18 (Final Path Fix Version) ---")
        self.arduino = ArduinoCommunicator(SER_PORT, SER_BAUD)
        self.language = LanguageModel(LANGUAGE_MODEL_FILE)
        self.services = Services()
        
        self.llm = Vertex_AI_Adapter(project=GCP_PROJECT_ID, location=GCP_REGION)
        
        self.face_processor = FaceProcessor()
        self.face_processor.start()
        print("--- Initialization complete, starting main loop ---")

    def run(self):
        try:
            while True:
                self.execute_state_logic()
                time.sleep(BRAIN_TICK_RATE)
        except KeyboardInterrupt:
            print("\nCtrl+C detected, shutting down.")
        finally:
            self.face_processor.stop()
            self.face_processor.join(timeout=2)
            self.arduino.send_command('s')
            self.arduino.send_command('p90')
            self.arduino.close()
            self.services.shutdown()
            print("Robot has been shut down safely.")

    def execute_state_logic(self):
        if not self.face_processor.is_alive() and self.state != "SYSTEM_ERROR":
            self.change_state("SYSTEM_ERROR")
        
        with self.face_processor.lock:
            threat_detected = self.face_processor.threat_detected
        if threat_detected and self.state != "ESCAPING":
            self.change_state("ESCAPING")
            return

        if self.state == "SYSTEM_ERROR":
            self.arduino.send_command('s')
            return

        elif self.state == "ESCAPING":
            self.arduino.send_command('b')
            self.services.speak(self.language.get_sentence("escape"))
            if not threat_detected: self.change_state("HUNTING_SEARCHING")
        
        elif self.state == "HUNTING_SEARCHING":
            self.arduino.send_command('l')
            if self.face_processor.any_person_detected:
                self.change_state("HUNTING_APPROACHING")

        elif self.state == "HUNTING_APPROACHING":
            self.handle_visual_servoing()

        elif self.state == "INTERACTING_ASK_FOOD":
            self.current_sentence = self.language.get_sentence("hunt")
            self.services.speak(self.current_sentence)
            self.change_state("INTERACTING_WAIT_FOOD")
        
        elif self.state == "INTERACTING_WAIT_FOOD":
            if self.arduino.read_and_reset_button_press():
                self.language.update_sentence_score("hunt", self.current_sentence, True)
                self.change_state("INTERACTING_SUCCESS_FOOD")
                return

            if self.check_interaction_timeout(): return
            
            reply = self.services.listen_for_speech(timeout=2)
            if reply:
                intent = self.llm.analyze_intent(reply)
                if intent == 'affirmative':
                    self.language.update_sentence_score("hunt", self.current_sentence, True)
                    self.change_state("INTERACTING_SUCCESS_FOOD")
                elif intent == 'negative':
                    self.language.update_sentence_score("hunt", self.current_sentence, False)
                    self.change_state("INTERACTING_FAIL")

        elif self.state == "INTERACTING_SUCCESS_FOOD":
            self.services.play_sound("happy")
            self.services.speak(self.language.get_sentence("success_food"))
            self.change_state("INTERACTING_ASK_REPRODUCE")
            
        elif self.state == "INTERACTING_ASK_REPRODUCE":
            self.current_sentence = self.language.get_sentence("reproduce")
            self.services.speak(self.current_sentence)
            self.change_state("INTERACTING_WAIT_REPRODUCE")

        elif self.state == "INTERACTING_WAIT_REPRODUCE":
            if self.check_interaction_timeout(): return
            
            reply = self.services.listen_for_speech(timeout=2)
            if reply:
                intent = self.llm.analyze_intent(reply)
                if intent == 'affirmative':
                    self.language.update_sentence_score("reproduce", self.current_sentence, True)
                    self.change_state("INTERACTING_SUCCESS_REPRODUCE")
                elif intent == 'negative':
                    self.language.update_sentence_score("reproduce", self.current_sentence, False)
                    self.change_state("INTERACTING_FAIL")
        
        elif self.state == "INTERACTING_SUCCESS_REPRODUCE":
            self.services.play_sound("happy")
            self.services.speak(self.language.get_sentence("success_reproduce"))
            self.change_state("HUNTING_SEARCHING")

        elif self.state == "INTERACTING_FAIL":
            self.services.play_sound("sad")
            self.services.speak(self.language.get_sentence("failure"))
            self.arduino.send_command('r')
            time.sleep(1.5)
            self.arduino.send_command('f')
            time.sleep(1.5)
            self.change_state("HUNTING_SEARCHING")

    def handle_visual_servoing(self):
        with self.face_processor.lock: bbox = self.face_processor.main_face_bbox
        if not bbox:
            self.change_state("HUNTING_SEARCHING")
            return

        (x, y, w, h) = bbox
        if h / FRAME_HEIGHT > TARGET_FACE_HEIGHT_RATIO:
            self.arduino.send_command('s')
            self.change_state("INTERACTING_ASK_FOOD")
            return

        now = time.time()
        face_center_x, face_center_y = x + w // 2, y + h // 2
        
        if now - self.last_move_command_time > MOVE_COMMAND_COOLDOWN:
            self.last_move_command_time = now
            target_x_min = FRAME_WIDTH * HORIZONTAL_TARGET_ZONE[0]
            target_x_max = FRAME_WIDTH * HORIZONTAL_TARGET_ZONE[1]
            if face_center_x < target_x_min: self.arduino.send_command('l')
            elif face_center_x > target_x_max: self.arduino.send_command('r')
            else: self.arduino.send_command('f')
        
        if now - self.last_tilt_command_time > TILT_COMMAND_COOLDOWN:
            self.last_tilt_command_time = now
            target_y_min = FRAME_HEIGHT * VERTICAL_TARGET_ZONE[0]
            target_y_max = FRAME_HEIGHT * VERTICAL_TARGET_ZONE[1]
            if face_center_y < target_y_min: self.arduino.send_command('u')
            elif face_center_y > target_y_max: self.arduino.send_command('d')

    def change_state(self, new_state):
        if self.state != new_state:
            print(f"State change: {self.state} -> {new_state}")
            self.state = new_state
            if "INTERACTING_WAIT" in new_state:
                self.interaction_start_time = time.time()

    def check_interaction_timeout(self):
        if time.time() - self.interaction_start_time > INTERACTION_TIMEOUT:
            print("[交互] 等待超时。")
            category_to_update = "hunt" if "FOOD" in self.state else "reproduce"
            self.language.update_sentence_score(category_to_update, self.current_sentence, False)
            self.change_state("INTERACTING_FAIL")
            return True
        return False

if __name__ == "__main__":
    # The main python file is named animal_brain.py, so update the warning message
    if GCP_PROJECT_ID == "your-gcp-project-id":
        print("\n\n[!!!] CRITICAL WARNING: You have not set up your Google Cloud Project ID.")
        print("The robot will NOT be able to understand voice commands. Please edit animal_brain.py file.\n\n")
    robot = AnimalRobot()
    robot.run()