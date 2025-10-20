from langchain_community.llms import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# Initialize endpoint for text generation
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",
    temperature=0.7,
    max_new_tokens=256,
)

prompt = "Generate 5 challenging DSA tasks for advanced learners."

response = llm.invoke(prompt)

print("Generated Tasks:\n", response)
