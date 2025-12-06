from abc import ABC, abstractmethod
import requests
import json

class ChatAdapter(ABC):
    @abstractmethod
    def send_message(self, message: str, **kwargs) -> str:
        pass

class GenericAPIAdapter(ChatAdapter):
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key

    def send_message(self, message: str, **kwargs) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Default payload structure - can be made configurable if needed
        payload = {"query": message}
        
        try:
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Attempt to extract answer from common keys
            data = response.json()
            if "answer" in data:
                return data["answer"]
            elif "response" in data:
                return data["response"]
            elif "message" in data:
                return data["message"]
            else:
                return str(data) # Fallback
                
        except requests.exceptions.RequestException as e:
            return f"Error connecting to server: {str(e)}"
        except json.JSONDecodeError:
            return f"Error decoding response: {response.text}"

class AdapterFactory:
    @staticmethod
    def get_adapter(mode: str, base_url: str, api_key: str = None) -> ChatAdapter:
        if mode == "Generic API":
            return GenericAPIAdapter(base_url, api_key)
        # Add other modes here (e.g. "Platform X")
        else:
            return GenericAPIAdapter(base_url, api_key)
