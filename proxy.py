from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv


import os
import json
import requests

load_dotenv()

bearer_token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')
flan_endpoint = os.environ.get('FLAN_HF_URL')
falcon_endpoint = os.environ.get('FALCON_HF_URL')
llama_endpoint = os.environ.get('LLAMA2_HF_URL')

class LLMPrompt(BaseModel):
    prompt: str


def proxyLLM(prompt: LLMPrompt, endpoint : str):
    print(f"Prompt: {prompt.prompt}")
    headers = {
        'Authorization': f'Bearer {bearer_token}',
        # Already added when you pass json=
        # 'Content-Type': 'application/json',
    }
    json_data = {
        'inputs': prompt.prompt,
    }
    response = requests.post(endpoint, headers=headers, json=json_data).json()[0]["generated_text"].strip()
    print(f"Response: {json.dumps(response, indent=4)}")
    return response




app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Try endpoints /flan , /falcon , or /llama2 "}


@app.post('/flan')
def proxyFlan(prompt: LLMPrompt):
    # Make a request to your large language model endpoint
    return proxyLLM(prompt, flan_endpoint)

@app.post('/falcon')
def proxyFalcon(prompt: LLMPrompt):
    # Make a request to your large language model endpoint
    return proxyLLM(prompt, falcon_endpoint)

@app.post('/llama2')
def proxyFalcon(prompt: LLMPrompt):
    # Make a request to your large language model endpoint
    return proxyLLM(prompt, llama_endpoint)



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)