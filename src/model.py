"""model.py
Model & tokenizer utilities using HuggingFace transformers.
Default: mBART large 50.
"""
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

MODEL_NAME = 'facebook/mbart-large-50-many-to-many-mmt'

def load_model_and_tokenizer(model_name: str = MODEL_NAME, src_lang: str = 'en_XX', tgt_lang: str = 'vi_VN'):
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    return model, tokenizer

if __name__ == '__main__':
    m, t = load_model_and_tokenizer()
    print('Loaded', t.vocab_size, 'tokens')
