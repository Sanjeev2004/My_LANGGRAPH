from langchain_huggingface import ChatHuggingFace
from dotenv import load_dotenv
import os

load_dotenv()
print("HF API Key Loaded:", os.getenv("HUGGINGFACEHUB_API_TOKEN") is not None)

model = ChatHuggingFace(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    max_new_tokens=128,
    temperature=0.7,
)

response = model.invoke("Hello! How are you today?")
print("Response:", response.content[0].text if hasattr(response, "content") else response)
