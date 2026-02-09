# Text Autocomplete — Нейросеть для автодополнения текстов

## Описание
Проект решает задачу автодополнения коротких текстовых постов. Сравниваются два подхода:
- **LSTM** — лёгкая рекуррентная модель, обученная с нуля
- **distilgpt2** — предобученная трансформерная модель

## Структура проекта
```
text-autocomplete/
├── data/                            # Датасеты
│   ├── raw_dataset.txt              # Сырой датасет
│   ├── dataset_processed.csv        # Очищенный датасет
│   ├── train.csv / val.csv / test.csv
├── src/                             # Код проекта
│   ├── data_utils.py                # Загрузка и очистка данных
│   ├── next_token_dataset.py        # Токенизатор, Dataset, DataLoader
│   ├── lstm_model.py                # LSTM модель
│   ├── lstm_train.py                # Обучение LSTM
│   ├── eval_lstm.py                 # Оценка LSTM
│   ├── eval_transformer_pipeline.py # Оценка distilgpt2
├── models/                          # Веса моделей
├── configs/                         # Конфигурации
├── solution.ipynb                   # Ноутбук с решением
└── requirements.txt                 # Зависимости
```

## Результаты

| Модель     | ROUGE-1 (val) | ROUGE-2 (val) | ROUGE-1 (test) | ROUGE-2 (test) |
|------------|---------------|---------------|----------------|----------------|
| LSTM       | 0.0353        | 0.0000        | —              | —              |
| distilgpt2 | 0.0537        | 0.0021        | 0.0616         | 0.0068         |

## Запуск
```bash
pip install -r requirements.txt
python3 src/data_utils.py
python3 src/lstm_train.py
python3 src/eval_lstm.py
python3 src/eval_transformer_pipeline.py
```
