# src/data_utils.py

import re
import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Загружает сырой датасет."""
    texts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
    df = pd.DataFrame({'text': texts})
    return df


def clean_text(text: str) -> str:
    """Очищает один текст."""
    # Удаляем @упоминания
    text = re.sub(r'@\w+', '', text)
    # Удаляем URL
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Приводим к нижнему регистру
    text = text.lower()
    # Оставляем только буквы, цифры, базовую пунктуацию и пробелы
    text = re.sub(r"[^a-z0-9\s\.\,\!\?\'\-]", '', text)
    # Убираем множественные пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_dataset(df: pd.DataFrame, min_words: int = 4, max_words: int = 30) -> pd.DataFrame:
    """Очищает датасет и фильтрует по длине."""
    df = df.copy()
    df['text'] = df['text'].apply(clean_text)
    # Убираем пустые строки
    df = df[df['text'].str.len() > 0]
    # Фильтруем по количеству слов (слишком короткие бесполезны, слишком длинные редки)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df = df[(df['word_count'] >= min_words) & (df['word_count'] <= max_words)]
    df = df.drop(columns=['word_count']).reset_index(drop=True)
    return df


def split_dataset(df: pd.DataFrame, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    """Разбивает датасет на train/val/test."""
    train_df, temp_df = train_test_split(df, test_size=(val_size + test_size), random_state=random_state)
    val_df, test_df = train_test_split(temp_df, test_size=test_size / (val_size + test_size), random_state=random_state)
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    return train_df, val_df, test_df


if __name__ == '__main__':
    # Загрузка и очистка
    df = load_raw_data('data/raw_dataset.txt')
    print(f"Исходный датасет: {len(df)} строк")
    
    df = preprocess_dataset(df)
    print(f"После очистки: {len(df)} строк")
    
  
    # Для GPU — используем весь датасет
    max_samples = None
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"Ограничено до: {len(df)} строк")
    
    # Сохраняем очищенный датасет
    df.to_csv('data/dataset_processed.csv', index=False)
    
    # Разбиваем на train/val/test
    train_df, val_df, test_df = split_dataset(df)
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/val.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Примеры:\n{train_df['text'].head(5).to_string()}")