from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# Log in to Hugging Face
login()

model_name = "meta-llama/Llama-2-7b-hf"

# Increase timeout
import requests
requests.adapters.DEFAULT_RETRIES = 5  # Retry 5 times

model = AutoModelForCausalLM.from_pretrained(model_name, timeout=300)  # Increase timeout to 300 sec
tokenizer = AutoTokenizer.from_pretrained(model_name, timeout=300)