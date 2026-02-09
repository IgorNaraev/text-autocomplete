# src/eval_transformer_pipeline.py

import sys
sys.path.insert(0, 'src')

import torch
import pandas as pd
from transformers import pipeline
from rouge_score import rouge_scorer
from tqdm import tqdm


def evaluate_transformer(val_path, max_samples=500):
    """Замер ROUGE для distilgpt2."""
    device_num = 0 if torch.cuda.is_available() else -1
    generator = pipeline("text-generation", model="distilgpt2", device=device_num)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)

    df = pd.read_csv(val_path)

    rouge1_scores = []
    rouge2_scores = []
    examples = []

    for i in tqdm(range(min(max_samples, len(df))), desc="distilgpt2 eval"):
        text = str(df.iloc[i]['text'])
        words = text.split()

        if len(words) < 4:
            continue

        split_point = int(len(words) * 3 / 4)
        prompt = ' '.join(words[:split_point])
        target = ' '.join(words[split_point:])
        num_target_tokens = len(target.split())

        try:
            result = generator(
                prompt,
                max_new_tokens=num_target_tokens + 5,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                temperature=0.8,
                pad_token_id=generator.tokenizer.eos_token_id
            )
            generated_full = result[0]['generated_text']
            generated_continuation = generated_full[len(prompt):].strip()
            generated_words = generated_continuation.split()[:num_target_tokens]
            generated_continuation = ' '.join(generated_words)
        except Exception as e:
            print(f"Error: {e}")
            continue

        if target and generated_continuation:
            scores = scorer.score(target, generated_continuation)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)

        if len(examples) < 10:
            examples.append({
                'input': prompt,
                'target': target,
                'generated': generated_continuation
            })

    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0

    return avg_rouge1, avg_rouge2, examples


if __name__ == '__main__':
    print("=== distilgpt2: Валидационная выборка ===")
    rouge1, rouge2, examples = evaluate_transformer('data/val.csv', max_samples=500)

    print(f"\nROUGE-1: {rouge1:.4f}")
    print(f"ROUGE-2: {rouge2:.4f}")

    print("\nПримеры предсказаний:")
    for ex in examples[:5]:
        print(f"  Вход:    {ex['input']}")
        print(f"  Таргет:  {ex['target']}")
        print(f"  Модель:  {ex['generated']}")
        print()