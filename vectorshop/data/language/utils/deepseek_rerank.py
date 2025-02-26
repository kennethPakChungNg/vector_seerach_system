import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import pandas as pd
from vectorshop.data.language.utils.performance import measure_performance

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

# Lazy-loaded model and tokenizer
_tokenizer = None
_model = None

def get_deepseek_model(device="cpu"):
    """
    Load DeepSeek model and tokenizer on the specified device if not already loaded.
    
    Args:
        device (str): Device to load the model on (default "cpu").
    
    Returns:
        tuple: Tokenizer and model.
    """
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map={"": device},  # Use specified device
            quantization_config=quantization_config,
            torch_dtype=torch.float16
        )
    return _tokenizer, _model

@measure_performance
def re_rank_results(query: str, results: pd.DataFrame, top_k: int = 5, device="cpu"):
    """
    Re-rank search results using the DeepSeek model's reasoning capability.
    
    Args:
        query (str): The search query.
        results (pd.DataFrame): Initial search results from FAISS.
        top_k (int): Number of results to return after re-ranking.
        device (str): Device to use for model computations (default "cpu").
    
    Returns:
        pd.DataFrame: Re-ranked results.
    """
    tokenizer, model = get_deepseek_model(device)
    re_ranked = []
    
    for idx, row in results.iterrows():
        text = row["cleaned_text"]
        prompt = f"<think>\nQuery: {query}\nText: {text}\nPlease output a relevance score between 0 and 100.\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=50, temperature=0.6, do_sample=False)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        match = re.search(r'(\d+)', response)
        score = int(match.group(1)) if match else 0
        re_ranked.append((row, score))
    
    re_ranked.sort(key=lambda x: x[1], reverse=True)
    top_results = [item[0] for item in re_ranked[:top_k]]
    return pd.DataFrame(top_results)