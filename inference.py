from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import re

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

def inference(query):
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    # PT
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    text = preprocess(query)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # Print labels and scores
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    ran = []
    for i in range(scores.shape[0]):
        l = config.id2label[ranking[i]]
        s = scores[ranking[i]]
        ran.append(s)

    return ran

def run_inference(query):
    return inference(query=query)

