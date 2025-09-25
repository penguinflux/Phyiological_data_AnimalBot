#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vertex AI Adapter
File: vertex_ai_adapter.py

描述:
这个模块取代了旧的 LLM_Communicator，使用直接的 REST API 请求
与 Google Cloud Vertex AI 上的 Gemini 模型进行通信。
这提供了比公共 Google AI Studio API 更低的延迟。

它还保留了本地的“小词典”以实现对简单命令的即时响应。
"""

import json
import requests
import google.auth
import google.auth.transport.requests

class Vertex_AI_Adapter:
    def __init__(self, project: str, location: str, model: str = "gemini-1.5-flash-preview-0514"):
        self.project = project
        self.location = location
        self.base_url = f"https://{location}-aiplatform.googleapis.com/v1"
        self.model_resource = f"projects/{project}/locations/{location}/publishers/google/models/{model}"
        self._token = None
        self.timeout = 15 # 15秒超时

        # 本地小词典，用于快速反应
        self.simple_affirmations = ['yes', 'yeah', 'yep', 'sure', 'okay', 'ok', 'fine', 'please']
        self.simple_negations = ['no', 'nope', "don't", 'stop']
        
        try:
            # 尝试在初始化时获取一次token，以便尽早发现认证问题
            self._refresh_token()
            print("[Vertex AI] 成功获取认证令牌。")
        except Exception as e:
            print(f"[Vertex AI][严重错误] 获取谷歌云认证失败: {e}")
            print("[Vertex AI] 请确保您已根据 README.md 完成了 gcloud CLI 的配置。")


    def _refresh_token(self):
        """刷新 OAuth 令牌。"""
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)
        self._token = creds.token

    def _ensure_token(self):
        """确保我们有一个有效的令牌，如果需要则刷新。"""
        # 简单起见，我们可以在每次调用前都刷新，或者实现更复杂的过期检查
        self._refresh_token()

    def analyze_intent(self, user_reply: str) -> str:
        """
        分析用户意图，优先使用本地词典，否则调用 Vertex AI。
        """
        if not user_reply:
            return "unclear"

        # 1. 检查本地小词典（快速路径）
        reply_lower = user_reply.lower().strip()
        if reply_lower in self.simple_affirmations:
            print("[Vertex AI] 本地词典匹配: 'affirmative'")
            return 'affirmative'
        if reply_lower in self.simple_negations:
            print("[Vertex AI] 本地词典匹配: 'negative'")
            return 'negative'

        # 2. 如果本地未命中，再使用 Vertex AI（慢速但更强大）
        print(f"[Vertex AI] 本地未命中，正在向 Vertex AI 发送请求: '{user_reply}'")
        
        try:
            self._ensure_token()
            url = f"{self.base_url}/{self.model_resource}:generateContent"
            
            prompt = (
                "Analyze the user's intent from the following sentence. "
                "The user is responding to a small robot's request. "
                "Possible intents are: 'affirmative', 'negative', 'unclear'. "
                "Your response MUST be ONLY ONE of these three words. Do not add any explanation.\n\n"
                f"User's sentence: \"{user_reply}\"\n\n"
                "Intent:"
            )

            body = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 5},
            }

            headers = {
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            }

            resp = requests.post(url, headers=headers, json=body, timeout=self.timeout)
            resp.raise_for_status() # 如果状态码是4xx或5xx，则抛出异常

            resp_json = resp.json()
            
            # 解析响应
            intent = resp_json["candidates"][0]["content"]["parts"][0]["text"].strip().lower()
            print(f"[Vertex AI] 收到分析结果: '{intent}'")
            if intent in ['affirmative', 'negative', 'unclear']:
                return intent
            return "unclear"
            
        except requests.exceptions.RequestException as e:
            print(f"[Vertex AI][错误] 网络请求失败: {e}")
            return "unclear"
        except Exception as e:
            print(f"[Vertex AI][错误] 解析或API调用时出错: {e}")
            return "unclear"
