# src/eval_lstm.py

import sys
sys.path.insert(0, 'src')

import torch
from next_token_dataset import create_dataloaders
from lstm_model import LSTMModel
from lstm_train import compute_rouge


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tokenizer, train_loader, val_loader, test_loader = create_dataloaders(
        'data/train.csv', 'data/val.csv', 'data/test.csv',
        batch_size=128, min_freq=5
    )

    model = LSTMModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    ).to(device)
    model.load_state_dict(torch.load('models/lstm_best.pt', map_location=device))

    print("\n=== LSTM: Валидационная выборка ===")
    rouge1, rouge2, examples = compute_rouge(
        model, val_loader, tokenizer, device, max_samples=500
    )
    print(f"ROUGE-1: {rouge1:.4f}")
    print(f"ROUGE-2: {rouge2:.4f}")

    print("\nПримеры предсказаний:")
    for ex in examples[:5]:
        print(f"  Вход:    {ex['input']}")
        print(f"  Таргет:  {ex['target']}")
        print(f"  Модель:  {ex['generated']}")
        print()

    print("\n=== LSTM: Тестовая выборка ===")
    rouge1_t, rouge2_t, examples_t = compute_rouge(
        model, test_loader, tokenizer, device, max_samples=500
    )
    print(f"ROUGE-1: {rouge1_t:.4f}")
    print(f"ROUGE-2: {rouge2_t:.4f}")

    print("\nПримеры предсказаний:")
    for ex in examples_t[:5]:
        print(f"  Вход:    {ex['input']}")
        print(f"  Таргет:  {ex['target']}")
        print(f"  Модель:  {ex['generated']}")
        print()