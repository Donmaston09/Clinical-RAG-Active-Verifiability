import requests
import json

def generate_content(prompt: str, provider: str, api_key: str) -> str:
    """
    Unified lightweight REST wrapper for multiple LLM providers.
    Uses native requests rather than heavy SDKs to preserve fast deployments.
    Returns the raw string output from the LLM.
    """
    if not api_key:
        return ""
    
    provider = provider.strip().lower()
    
    try:
        if provider == "gemini":
            # Gemini REST API
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
        elif provider == "openai":
            # OpenAI REST API
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}]
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
        elif provider == "anthropic":
            # Anthropic REST API
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            payload = {
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 2048,
                "messages": [{"role": "user", "content": prompt}]
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            # Anthropic returns a list of content blocks
            content_blocks = data.get("content", [])
            if content_blocks:
                return content_blocks[0].get("text", "")
            return ""
            
        else:
            return ""
            
    except Exception as e:
        # We raise the exception so the caller can log or fallback
        # e.g., capturing 429 quota errors
        raise RuntimeError(f"{provider.capitalize()} API Error: {str(e)}")
