import openai
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from .constants import *

class LLM():
    def __init__(self, num_groups, num_stores):
        openai.api_key = API_KEY
        openai.api_type = API_TYPE
        openai.api_base = API_BASE
        openai.api_version = API_VERSION

        self.num_groups = num_groups
        self.num_stores = num_stores

    def model(self, message, max_tokens=128):
        def run_model(deployment_name, message, max_tokens):
            response = openai.ChatCompletion.create(
                    engine=deployment_name, 
                    messages=message, 
                    max_tokens=max_tokens,
                    temperature=0
                )
            return response
        
        while True:
            try:
                response = run_model(DEPLOYMENT_NAME, message, max_tokens)
                break
            except:
                try:
                    response = run_model(DEPLOYMENT_NAME_BACKUP, message, max_tokens)
                    break
                except:
                    print('Pause for 5 seconds and try again...')
                    import time
                    time.sleep(5)
        return response["choices"][0]["message"]["content"]
    
    def generate(self, prompt):
        message = [{"role": "user", "content": prompt}]
        out = self.model(message=message, max_tokens=max(len(prompt)*2, len(prompt)+128))
        out.replace('"', '').replace("'", '')
        return out
    
    def generate_responses_parallel(self, prompts):
        with ThreadPoolExecutor(max_workers=self.num_groups + self.num_stores + 10) as executor:
            futures = {executor.submit(self.generate, prompt): prompt for prompt in prompts}
            results = []
            for f in tqdm(futures):
                results.append(f.result())
        return results