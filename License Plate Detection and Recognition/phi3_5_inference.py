import os
import base64
import requests
import json

class TextExtractionModel_Phi_3_Model: 
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url

    def phi_3_inference(self, image_path):
        stream = False  # Set to True if you want a streaming response

        # ✅ Read image and encode in base64
        try:
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()
        except FileNotFoundError:
            print(f"[ERROR] Image file not found: {image_path}")
            return {"error": "File not found"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream" if stream else "application/json"
        }

        payload = {
            "model": 'microsoft/phi-3.5-vision-instruct',
            "messages": [
                {
                    "role": "user",
                    "content": f'Extract only the text from the image/document and return it exactly as it appears, without any additional words, explanations, or formatting.<img src="data:image/png;base64,{image_b64}" />'
                }
            ],
            "max_tokens": 512,
            "temperature": 0.20,
            "top_p": 0.70,
            "stream": stream
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            
            # ✅ Check if response is successful
            if response.status_code != 200:
                print(f"[ERROR] API request failed with status {response.status_code}: {response.text}")
                return {"error": response.text}

            # ✅ Handle streaming response
            if stream:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode("utf-8")
                        print(decoded_line)
                return {"message": "Streaming response received"}

            # ✅ Parse JSON response safely
            response_json = response.json()
            extracted_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")

            if not extracted_text:
                print("[WARNING] No extracted data found in response.")
                return {"error": "No text extracted"}

            print("[INFO] Extracted text:", extracted_text)
            return {"extracted_data": extracted_text}

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request failed: {e}")
            return {"error": str(e)}
