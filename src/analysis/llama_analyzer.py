import requests
import base64
import os
from dotenv import load_dotenv
import logging
from pathlib import Path
import backoff
import json

load_dotenv()

class LlamaAnalyzer:
    def __init__(self):
        self.api_key = os.getenv('TOGETHER_API_KEY')
        if not self.api_key:
            self.logger.warning("⚠️ TOGETHER_API_KEY not found in environment variables")
        self.model = "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"
        self.logger = logging.getLogger(__name__)
        
    def encode_image(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"❌ Failed to encode image {image_path}: {e}")
            return None
            
    @backoff.on_exception(
        backoff.expo,
        (json.JSONDecodeError, requests.exceptions.RequestException),
        max_tries=3,
        giveup=lambda e: isinstance(e, ValueError)
    )
    def analyze_fire(self, image_path):
        """Analyze fire severity using Llama Vision"""
        try:
            encoded_image = self.encode_image(image_path)
            if not encoded_image:
                return {'status': 'error', 'error': 'Failed to encode image'}

            prompt = """You are a fire detection AI assistant. Analyze the image and respond with a JSON object.

            RESPONSE FORMAT:
            {
                "fire_detected": boolean,
                "analysis": {
                    "severity": "none|low|medium|high|critical",
                    "spread_direction": string or null,
                    "risks": string[] or null,
                    "recommended_actions": string[] or null,
                    "emergency_services_required": boolean
                },
                "confidence": number (0-1)
            }

            WHAT NOT TO DO:
            1. DO NOT classify warm lighting or orange walls as fire
            2. DO NOT assume smoke from any source is fire
            3. DO NOT treat reflections or glare as flames
            4. DO NOT classify electronic displays or lights as fire
            5. DO NOT mark indoor lighting as potential fire sources

            WHAT TO DO:
            1. Look for active flames with characteristic flicker
            2. Check for visible smoke with clear source
            3. Verify thermal patterns typical of combustion
            4. Consider context (indoor/outdoor, natural/artificial light)
            5. Set emergency_services_required=true only for confirmed fires

            CONFIDENCE SCORING:
            - 1.0: Absolutely certain of assessment
            - 0.8-0.9: Clear visual evidence
            - 0.5-0.7: Some uncertainty
            - <0.5: Significant doubt

            If no fire is present, respond with:
            {
                "fire_detected": false,
                "analysis": {
                    "severity": "none",
                    "spread_direction": null,
                    "risks": null,
                    "recommended_actions": null,
                    "emergency_services_required": false
                },
                "confidence": 1.0
            }"""
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "temperature": 0.7,
                "max_tokens": 512,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }]
            }
            
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                analysis = response.json()['choices'][0]['message']['content']
                return {
                    'status': 'success',
                    'analysis': analysis
                }
            else:
                raise Exception(f"API Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }