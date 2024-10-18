import os
from dotenv import load_dotenv
from langchain.llms import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from the environment variables

os.environ["OPENAI_API_KEY"]="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"
# Initialize the OpenAI LLM with the API key
llm = OpenAI()

# Test prompt
prompt = "Explain what sustainability is."

# Generate a response using the LLM
response = llm.predict(prompt)

# Print the response
print("Response from OpenAI:", response)
