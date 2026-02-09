# src/lstm_train.py

import sys
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from rouge_score import rouge_scorer
from tqdm import tqdm

sys.path.insert(0, 'src')
from next_token_dataset import create_dataloaders
from lstm_model import LSTMModel


def compute_rouge(model, dataloader, tokenizer, device, max_samples=500):
    """Замеряет ROUGE на выборке. Вход: 3/4 текста, таргет: последняя 1/4."""
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    examples = []
    
    sample_count = 0
    
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            for i in range(x_batch.size(0)):
                if sample_count >= max_samples:
                    break
                
                # Полная последовательность (x без паддинга)
                tokens = x_batch[i].tolist()
                # Убираем паддинг
                tokens = [t for t in tokens if t != 0]
                
                if len(tokens) < 4:
                    continue
                
                # 3/4 — вход, 1/4 — таргет
                split_point = int(len(tokens) * 3 / 4)
                if split_point < 1:
                    split_point = 1
                
                input_tokens = tokens[:split_point]
                target_tokens = tokens[split_point:]
                
                # Генерируем
                prompt_text = tokenizer.decode(input_tokens)
                num_to_generate = len(target_tokens)
                generated_text = model.generate(
                    tokenizer, prompt_text,
                    max_new_tokens=num_to_generate, device=device
                )
                
                # Убираем промпт из сгенерированного текста
                generated_continuation = generated_text[len(prompt_text):].strip()
                target_text = tokenizer.decode(target_tokens)
                
                if target_text and generated_continuation:
                    scores = scorer.score(target_text, generated_continuation)
                    rouge1_scores.append(scores['rouge1'].fmeasure)
                    rouge2_scores.append(scores['rouge2'].fmeasure)
                
                # Сохраняем примеры
                if len(examples) < 5:
                    examples.append({
                        'input': prompt_text,
                        'target': target_text,
                        'generated': generated_continuation
                    })
                
                sample_count += 1
            
            if sample_count >= max_samples:
                break
    
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
    
    return avg_rouge1, avg_rouge2, examples


def train_model(
    model, train_loader, val_loader, tokenizer, device,
    epochs=10, lr=0.001, save_path='models/lstm_best.pt'
):
    """Обучение модели."""
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # игнорируем паддинг
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'rouge1': [], 'rouge2': []}
    
    for epoch in range(epochs):
        # === TRAIN ===
        model.train()
        total_train_loss = 0
        num_batches = 0
        start_time = time.time()
        
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(x_batch)
            
            # logits: (batch, seq, vocab) -> (batch*seq, vocab)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y_batch.reshape(-1))
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_train_loss / num_batches
        epoch_time = time.time() - start_time
        
        # === VALIDATION LOSS ===
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for x_batch, y_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits, _ = model(x_batch)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y_batch.reshape(-1))
                total_val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = total_val_loss / num_val_batches
        
        # === ROUGE (каждые 2 эпохи для ускорения) ===
        rouge1, rouge2, examples = 0.0, 0.0, []
        if (epoch + 1) % 2 == 0 or epoch == 0 or (epoch + 1) == epochs:
            rouge1, rouge2, examples = compute_rouge(
                model, val_loader, tokenizer, device, max_samples=300
            )
        
        # Сохраняем историю
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['rouge1'].append(rouge1)
        history['rouge2'].append(rouge2)
        
        print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.0f}s)")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  ROUGE-1: {rouge1:.4f} | ROUGE-2: {rouge2:.4f}")
        
        if examples:
            print("\n  Примеры предсказаний:")
            for ex in examples[:3]:
                print(f"    Вход: {ex['input']}")
                print(f"    Таргет: {ex['target']}")
                print(f"    Модель: {ex['generated']}")
                print()
        
        # Сохраняем лучшую модель
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  Модель сохранена ({save_path})")
    
    return history


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
    
    print(f"Параметры модели: {sum(p.numel() for p in model.parameters()):,}")
    
    history = train_model(
        model, train_loader, val_loader, tokenizer, device,
        epochs=10, lr=0.001, save_path='models/lstm_best.pt'
    )