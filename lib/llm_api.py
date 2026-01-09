import os
import yaml
import random
import time
from litellm import Router
from dotenv import load_dotenv

load_dotenv("../.env")

def initialize_router():
    try:
        with open("../litellm.yaml", "r") as f:
            config = yaml.safe_load(os.path.expandvars(f.read()))
        return Router(model_list=config.get("model_list", []))
    except FileNotFoundError:
        print("エラー: litellm.yaml が見つかりません。")
        exit(1)

def completion_with_backoff(**kwargs):
    """
    RateLimitError発生時に指数関数的バックオフでリトライを行うラッパー関数
    """
    router = kwargs.pop("router", None)
    if not router:
        raise ValueError("router is a required keyword argument")

    max_retries = 8
    base_delay = 5
    
    if "timeout" not in kwargs:
        kwargs["timeout"] = 120
    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = 8192

    for attempt in range(max_retries + 1):
        try:
            return router.completion(**kwargs)
        except Exception as e:
            error_str = str(e)
            if ("RateLimitError" in error_str or 
                "429" in error_str or 
                "ServiceUnavailableError" in error_str or 
                "Timeout" in error_str):
                
                if attempt < max_retries:
                    delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    print(f"  [Retry] エラー発生: {e}")
                    print(f"  -> {delay:.2f}秒待機してリトライします... (試行 {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
            
            raise e
