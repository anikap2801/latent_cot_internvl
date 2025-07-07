from transformers import AutoTokenizer

def get_tokenizer():
    return AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B-Pretrained", trust_remote_code=True)
