from transformers import AutoTokenizer
import numpy as np
import torch
import re
from typing import TypedDict
import model_longformer

class DetectorOutput(TypedDict):
    real: float
    fake: float

INFERENCE_MODEL = model_longformer.Model.from_ckpt("longformer-flask-ckpt.pth")

def replace_substring(test_str, s1, s2):
    # Replacing all occurrences of substring s1 with s2
    test_str = re.sub(s1, s2, test_str)
    return test_str

def preprocess(text):
    new_text = []

    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)

    return " ".join(new_text)

def inference(query) -> DetectorOutput:
    model_tag = f"allenai/longformer-base-4096"
    tokenizer = AutoTokenizer.from_pretrained(model_tag, model_max_length=4096)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True  # type: ignore

    text = preprocess(query)
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    output = INFERENCE_MODEL.model.forward(**encoded_input, return_dict=True)

    out_logits = output.logits
    out_conf = torch.nn.functional.softmax(out_logits, dim=-1)
    out_conf = out_conf.detach().numpy()

    return {
        "real": float(out_conf[0][1]),
        "fake": float(out_conf[0][0])
    }

def run_inference(query):
    # TODO: why is this wrapper just here?
    return inference(query=query)
