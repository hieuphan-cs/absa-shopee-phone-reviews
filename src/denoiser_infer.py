from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List


def load_denoiser(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    return tokenizer, model


def denoise_texts(texts: List[str], tokenizer, model, device='cpu', max_length=128):
    model.to(device)
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
    outs = model.generate(**inputs, max_length=max_length, num_beams=4)
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outs]


if __name__ == '__main__':
    # example usage
    tok, m = load_denoiser('outputs/denoiser')
    out = denoise_texts(["đt đep qua z :))"], tok, m)
    print(out)
