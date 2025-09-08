"""translate.py
Simple script to translate input sentences with a saved model.
"""
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--sentence', type=str, required=True)
    parser.add_argument('--max_length', type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    if hasattr(tokenizer, 'src_lang') and hasattr(tokenizer, 'tgt_lang'):
        tokenizer.src_lang = 'en_XX'
        tokenizer.tgt_lang = 'vi_VN'
    inputs = tokenizer(args.sentence, return_tensors='pt', truncation=True)
    outputs = model.generate(**inputs, max_length=args.max_length, num_beams=4)
    print('Prediction:', tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == '__main__':
    main()
