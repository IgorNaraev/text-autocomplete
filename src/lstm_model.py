# src/lstm_model.py

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """LSTM модель для предсказания следующего токена."""
    
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        """
        Forward pass для обучения.
        x: (batch_size, seq_len)
        Возвращает: logits (batch_size, seq_len, vocab_size)
        """
        emb = self.embedding(x)            # (batch, seq_len, embed_dim)
        out, hidden = self.lstm(emb, hidden)  # (batch, seq_len, hidden_dim)
        out = self.dropout(out)
        logits = self.fc(out)               # (batch, seq_len, vocab_size)
        return logits, hidden
    
    def generate(self, tokenizer, prompt_text, max_new_tokens=10, device='cpu', temperature=0.8):
        """
        Генерация нескольких токенов по начальному тексту.
        """
        self.eval()
        tokens = tokenizer.encode(prompt_text)[:-1]  # убираем <eos> из промпта
        input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
        
        # Индексы токенов, которые нельзя генерировать
        pad_idx = tokenizer.word2idx[tokenizer.pad_token]
        eos_idx = tokenizer.word2idx[tokenizer.eos_token]
        unk_idx = tokenizer.word2idx[tokenizer.unk_token]
        
        hidden = None
        generated = list(tokens)
        
        with torch.no_grad():
            # Прогоняем промпт через модель
            logits, hidden = self.forward(input_ids, hidden)
            
            for i in range(max_new_tokens):
                next_logits = logits[:, -1, :] / temperature
                
                # Запрещаем pad и eos
                next_logits[:, pad_idx] = float('-inf')
                next_logits[:, eos_idx] = float('-inf')
                next_logits[:, unk_idx] = float('-inf')
                
                # Sampling
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                generated.append(next_token)
                
                inp = torch.tensor([[next_token]], dtype=torch.long).to(device)
                logits, hidden = self.forward(inp, hidden)
        
        return tokenizer.decode(generated)


if __name__ == '__main__':
    from next_token_dataset import create_dataloaders
    
    tokenizer, train_loader, val_loader, test_loader = create_dataloaders(
        'data/train.csv', 'data/val.csv', 'data/test.csv'
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = LSTMModel(vocab_size=tokenizer.vocab_size).to(device)
    print(f"Параметры модели: {sum(p.numel() for p in model.parameters()):,}")
    
    # Проверяем forward на одном батче
    x_batch, y_batch = next(iter(train_loader))
    x_batch = x_batch.to(device)
    logits, _ = model(x_batch)
    print(f"Logits shape: {logits.shape}")
    
    # Проверяем генерацию (необученная модель — ерунда, но не должно падать)
    result = model.generate(tokenizer, "i love this", max_new_tokens=5, device=device)
    print(f"Генерация (необученная): {result}")