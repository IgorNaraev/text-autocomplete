# Text Autocomplete — Нейросеть для автодополнения текстов

## Описание

Проект решает задачу автодополнения коротких текстовых постов. Сравниваются два подхода:

- **LSTM** — лёгкая рекуррентная модель (~15M параметров), обученная с нуля на датасете sentiment140
- **distilgpt2** — предобученная трансформерная модель (~82M параметров), используется без дообучения

**Датасет:** sentiment140 (~1.6M коротких текстовых постов)

**Задача:** по началу текста (3/4) предсказать продолжение (1/4)

**Метрики:** ROUGE-1, ROUGE-2

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
|------------|---------------|---------------|-----------------|-----------------|
| LSTM       | 0.0550        | 0.0073        | 0.0638          | 0.0057          |
| distilgpt2 | 0.0554        | 0.0036        | 0.0714          | 0.0051          |

Модели показывают сопоставимые результаты. Для мобильного приложения рекомендуется LSTM благодаря меньшему размеру (~15M vs ~82M параметров).

## Запуск
```bash
pip install -r requirements.txt
python3 src/data_utils.py
python3 src/lstm_train.py
python3 src/eval_lstm.py
python3 src/eval_transformer_pipeline.py
```
