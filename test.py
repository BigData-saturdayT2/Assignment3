from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Initialize ChatNVIDIA model
llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1", max_tokens=1024)

# Define a prompt to test
prompt = "Explain the concept of RAG in simple terms."

# Attempt to get a response using `invoke`
try:
    response = llm.invoke(prompt)  # Directly pass the prompt as a string
    print("API Test Response:", response)
except Exception as e:
    print("Error during NVIDIA API call:", e)
