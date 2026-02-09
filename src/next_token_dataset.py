# src/next_token_dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from collections import Counter


class Tokenizer:
    """Простой токенизатор на уровне слов."""
    
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
        # Специальные токены
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.eos_token = '<eos>'
    
    def build_vocab(self, texts: list):
        """Строит словарь по текстам."""
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        
        # Специальные токены
        special_tokens = [self.pad_token, self.unk_token, self.eos_token]
        self.word2idx = {token: idx for idx, token in enumerate(special_tokens)}
        
        # Добавляем слова с частотой >= min_freq
        idx = len(special_tokens)
        for word, freq in counter.most_common():
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                idx += 1
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
    
    def encode(self, text: str) -> list:
        """Преобразует текст в список индексов."""
        unk_idx = self.word2idx[self.unk_token]
        eos_idx = self.word2idx[self.eos_token]
        tokens = [self.word2idx.get(w, unk_idx) for w in text.split()]
        tokens.append(eos_idx)
        return tokens
    
    def decode(self, indices: list) -> str:
        """Преобразует список индексов обратно в текст."""
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, self.unk_token)
            if word == self.eos_token:
                break
            if word == self.pad_token:
                continue
            words.append(word)
        return ' '.join(words)


class NextTokenDataset(Dataset):
    """Dataset для задачи предсказания следующего токена."""
    
    def __init__(self, texts: list, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) >= 3:  # минимум 2 токена + eos
                self.data.append(torch.tensor(tokens, dtype=torch.long))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        x = tokens[:-1]  # входная последовательность
        y = tokens[1:]   # таргет (сдвиг на 1 вправо)
        return x, y


def collate_fn(batch):
    """Паддинг батча до одинаковой длины."""
    xs, ys = zip(*batch)
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0)
    ys_padded = pad_sequence(ys, batch_first=True, padding_value=0)
    return xs_padded, ys_padded


def create_dataloaders(train_path, val_path, test_path, batch_size=256, min_freq=2):
    """Создаёт токенизатор и DataLoader'ы."""
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # Строим словарь ТОЛЬКО по трейну
    tokenizer = Tokenizer(min_freq=min_freq)
    tokenizer.build_vocab(train_df['text'].tolist())
    print(f"Размер словаря: {tokenizer.vocab_size}")
    
    # Создаём датасеты
    train_dataset = NextTokenDataset(train_df['text'].tolist(), tokenizer)
    val_dataset = NextTokenDataset(val_df['text'].tolist(), tokenizer)
    test_dataset = NextTokenDataset(test_df['text'].tolist(), tokenizer)
    
    print(f"Train samples: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Создаём DataLoader'ы
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return tokenizer, train_loader, val_loader, test_loader


if __name__ == '__main__':
    tokenizer, train_loader, val_loader, test_loader = create_dataloaders(
        'data/train.csv', 'data/val.csv', 'data/test.csv'
    )
    
    # Проверяем один батч
    x_batch, y_batch = next(iter(train_loader))
    print(f"\nBatch X shape: {x_batch.shape}, Y shape: {y_batch.shape}")
    
    # Пример кодирования/декодирования
    sample_text = "i love this movie so much"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    print(f"\nОригинал: {sample_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")