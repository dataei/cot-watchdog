# do pip install -r requirements.txt
# huggingface-cli login
# python app.py
#then do:
#git add .
# git commit -m "Initial Nemotron project setup"
# git branch -M main
# git remote add origin git@github.com:YOUR_USERNAME/my-nemotron-project.git
# git push -u origin main

#first touch nemotron smoke test
#run first, confirms the model loads and generates text
#if this works move onto setup_checkpoint.py for full 
#validation suite
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
MODEL_ID = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
prompt = "Explain what this project does in one sentence."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=100
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))