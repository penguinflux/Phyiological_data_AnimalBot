#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NB3 Sound Adapter (Synchronous Fix Version)
File: nb3_sound_adapter.py

描述:
这是声音适配器的最终稳定版。
1.  所有音频播放都已改回同步（阻塞）模式，以确保交互的自然流程。
    机器人会一直等待，直到话说完或音效播放完毕。
2.  在 listen() 函数中，增加了将录音从立体声转换为单声道的关键步骤，
    以解决语音识别“听不懂”的问题。
"""

import time
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
import io
import speech_recognition as sr

# 尝试导入您专有的NB3声音库的特定模块
try:
    import NB3.Sound.utilities as Utilities
    import NB3.Sound.speaker as Speaker
    import NB3.Sound.microphone as Microphone
    print("[声音适配器] 成功导入 'NB3.Sound' 模块。")
    nb_sound_available = True
except ImportError:
    print("[声音适配器][严重警告] 无法导入 'NB3.Sound' 模块。")
    nb_sound_available = False

class NB3_Sound_Adapter:
    def __init__(self, sample_rate=48000, num_channels=2):
        self.speaker = None
        self.microphone = None
        self.recognizer = sr.Recognizer()
        self.sample_rate = sample_rate
        self.num_channels = num_channels 
        self.audio_format_str = 'int32'
        self.sample_width_bytes = 4 # int32 = 4 bytes

        if nb_sound_available:
            try:
                output_device = Utilities.get_output_device_by_name("MAX")
                speaker_buffer = int(self.sample_rate / 10)
                self.speaker = Speaker.Speaker(output_device, self.num_channels, self.audio_format_str, self.sample_rate, speaker_buffer)
                self.speaker.start()
                print(f"[声音适配器] NB3 Speaker 初始化成功 (格式: {self.audio_format_str})。")
                
                input_device = Utilities.get_input_device_by_name("MAX")
                mic_buffer = int(self.sample_rate / 10)
                mic_max_samples = int(self.sample_rate * 10)
                self.microphone = Microphone.Microphone(input_device, self.num_channels, self.audio_format_str, self.sample_rate, mic_buffer, mic_max_samples)
                self.microphone.start()
                print("[声音适配器] NB3 Microphone 初始化成功。")
            except Exception as e:
                print(f"[声音适配器][错误] 初始化 NB3 音频设备失败: {e}")
                self.speaker = None; self.microphone = None
    
    def speak(self, text, lang='en'):
        """同步语音合成与播放 (使用浮点数)"""
        if self.speaker:
            try:
                # 1. gTTS -> MP3 in memory
                mp3_fp = io.BytesIO()
                tts = gTTS(text=text, lang=lang)
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0)

                # 2. pydub: MP3 -> AudioSegment
                audio = AudioSegment.from_file(mp3_fp, format="mp3")
                
                audio = audio.set_frame_rate(self.sample_rate)
                audio = audio.set_channels(self.num_channels)

                # 3. pydub -> int16 numpy array
                samples_int16 = np.array(audio.get_array_of_samples())
                
                # 4. 将 int16 样本转换为 [-1.0, 1.0] 范围内的浮点数
                samples_float = samples_int16.astype(np.float32) / 32767.0
                
                if self.num_channels > 1:
                    samples_float = samples_float.reshape((-1, self.num_channels))
                
                # 5. 写入数据，然后等待播放完成
                self.speaker.write(samples_float)
                while self.speaker.is_playing():
                    time.sleep(0.05)

            except Exception as e:
                print(f"[声音适配器][错误] speak 功能失败: {e}")

    def _generate_tone_array(self, frequency, duration_ms, volume=0.7):
        """使用Numpy生成单个音调的浮点波形数组"""
        t = np.linspace(0., duration_ms / 1000., int(self.sample_rate * duration_ms / 1000.), endpoint=False)
        waveform_float = (volume * np.sin(2. * np.pi * frequency * t)).astype(np.float32)
        
        if self.num_channels > 1:
            data_stereo = np.zeros((len(waveform_float), self.num_channels), dtype=np.float32)
            for i in range(self.num_channels): data_stereo[:, i] = waveform_float
            return data_stereo
        return waveform_float

    def play_sound(self, sound_type):
        """同步音效播放 (使用浮点数)"""
        if self.speaker:
            try:
                if sound_type == "happy":
                    sound_array = np.concatenate((self._generate_tone_array(523, 150), self._generate_tone_array(784, 200)))
                elif sound_type == "sad":
                    sound_array = np.concatenate((self._generate_tone_array(392, 200), self._generate_tone_array(261, 250)))
                else: return
                
                self.speaker.write(sound_array)
                while self.speaker.is_playing():
                    time.sleep(0.05)
            except Exception as e:
                print(f"[声音适配器][错误] play_sound 功能失败: {e}")

    def listen(self, timeout=5):
        """使用NB3麦克风录音，并用SpeechRecognition库进行语音转文本"""
        if not self.microphone: return None

        print(f"[声音适配器] 开始录音 {timeout} 秒...")
        self.microphone.reset()
        time.sleep(timeout)
        
        num_samples = self.microphone.valid_samples
        if num_samples == 0:
            print("[声音适配器] 没有录到任何声音。")
            return None
            
        recording_data_stereo = self.microphone.latest(num_samples)
        print(f"[声音适配器] 录音结束，录到 {num_samples} 个立体声采样点。")

        try:
            # --- [代码修正] ---
            # 1. 将立体声转换为单声道 (取左声道)
            #    这是解决“听不懂”问题的关键
            recording_data_mono = recording_data_stereo[:, 0].copy()

            # 2. 将单声道数据转换为 AudioData 对象
            audio_data = sr.AudioData(recording_data_mono.tobytes(), self.sample_rate, self.sample_width_bytes)
            
            print("[声音适配器] 正在将语音转换为文本...")
            text = self.recognizer.recognize_google(audio_data, language='en-US')
            print(f"[声音适配器] 识别结果: '{text}'")
            return text
        except sr.UnknownValueError:
            print("[声音适配器] Google Speech Recognition 无法理解音频。")
            return None
        except sr.RequestError as e:
            print(f"[声音适配器] 无法从Google Speech Recognition服务请求结果; {e}")
            return None
        except Exception as e:
            print(f"[声音适配器][错误] 语音识别过程中出现未知错误: {e}")
            return None

    def shutdown(self):
        if self.speaker:
            try: self.speaker.stop()
            except: pass
        if self.microphone:
            try: self.microphone.stop()
            except: pass
        print("[声音适配器] NB3 音频设备已关闭。")

