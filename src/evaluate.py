"""evaluate.py
Evaluate a saved seq2seq model on the validation set (must exist).
Computes: SacreBLEU, METEOR, ROUGE-L, ChrF
"""
import argparse
import os
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dataset import prepare_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--max_length', type=int, default=128)
    args = parser.parse_args()

    ds = prepare_dataset(args.data_dir)
    if 'validation' not in ds:
        raise SystemExit('validation split not found in data/ (provide valid.en and valid.vi)')

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)

    sacrebleu = evaluate.load('sacrebleu')
    chrf = evaluate.load('chrf')
    meteor = evaluate.load('meteor')
    rouge = evaluate.load('rouge')

    refs = []
    hyps = []
    for item in ds['validation']:
        input_text = item['en']
        ref = item['vi']
        encoded = tokenizer(input_text, return_tensors='pt', truncation=True)
        generated = model.generate(**encoded, max_length=args.max_length, num_beams=4)
        pred = tokenizer.decode(generated[0], skip_special_tokens=True)
        hyps.append(pred)
        refs.append(ref)

    bleu = sacrebleu.compute(predictions=hyps, references=[[r] for r in refs])
    ch = chrf.compute(predictions=hyps, references=refs)
    me = meteor.compute(predictions=hyps, references=refs)
    ro = rouge.compute(predictions=hyps, references=refs)

    # Print standardized results
    print('SacreBLEU:', round(bleu.get('score', 0), 2))
    print('ChrF:', round(ch.get('score', 0), 2))
    # meteor returns {'meteor': value}
    print('METEOR:', round(me.get('meteor', 0), 2))
    # rouge returns keys like rouge1, rouge2, rougeL; we print rougeL f-measure
    rougeL = ro.get('rougeL', {})
    rougeL_f = rougeL.get('fmeasure', None) if isinstance(rougeL, dict) else None
    if rougeL_f is not None:
        print('ROUGE-L (F1):', round(rougeL_f * 100, 2))
    else:
        # some versions return percentages or different keys
        # try some fallbacks
        if 'rougeL' in ro and isinstance(ro['rougeL'], (int, float)):
            print('ROUGE-L:', round(ro['rougeL'], 2))
        else:
            print('ROUGE-L:', ro)

if __name__ == '__main__':
    main()
