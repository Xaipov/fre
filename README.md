# Fiverr Python Jobs Finder

Программа для автоматического поиска заказов на Fiverr, связанных с разработкой на Python.

## Возможности

- 🔍 **Автоматический поиск** заказов по ключевым словам через браузер
- 🧠 **Опциональная нейросеть** (до 4B параметров) для оценки релевантности заказов
- 💾 **Сохранение результатов** в JSON и TXT форматах
- 🎯 **Фильтрация** по релевантности разработке на Python

## Установка зависимостей

```bash
pip install selenium
```

### Опционально: для использования нейросети

```bash
pip install transformers torch accelerate
```

## Требования

1. **Google Chrome** должен быть установлен в системе
2. **ChromeDriver** должен соответствовать версии вашего Chrome

### Установка ChromeDriver

**Linux:**
```bash
sudo apt-get update
sudo apt-get install chromium-chromedriver
```

**Windows:**
Скачайте с https://chromedriver.chromium.org/downloads

**macOS:**
```bash
brew install chromedriver
```

## Использование

### Базовый запуск (без нейросети)

```bash
python fiverr_python_finder.py
```

### С использованием нейросети

Откройте файл `fiverr_python_finder.py` и измените настройки в функции `main()`:

```python
USE_NEURAL = True  # Включить нейросеть
```

Затем запустите:

```bash
python fiverr_python_finder.py
```

## Настройки

В начале функции `main()` вы можете изменить следующие параметры:

```python
USE_NEURAL = False  # Использовать ли нейросеть (True/False)
HEADLESS = False    # Запускать браузер без интерфейса (True/False)
MAX_JOBS = 20       # Максимальное количество заказов для сбора
```

## Ключевые слова для поиска

По умолчанию программа ищет по следующим запросам:
- python development
- python developer needed
- python script
- python automation
- python bot telegram
- python web scraping
- python django developer
- python flask
- python api development
- python data scraping

Вы можете изменить список в коде в функции `main()`.

## Нейросеть

Программа поддерживает локальные модели до 4B параметров. По умолчанию используется:
- **Qwen/Qwen2.5-3B-Instruct** (3B параметров)

Для изменения модели отредактируйте класс `NeuralFilter`:

```python
def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
```

Другие подходящие модели:
- `microsoft/phi-2` (2.7B)
- `google/gemma-2b-it` (2B)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B)

## Результаты

Программа создает два файла:

1. **JSON файл** (`fiverr_python_jobs_YYYYMMDD_HHMMSS.json`) - полные данные в структурированном формате
2. **TXT файл** (`fiverr_python_jobs_YYYYMMDD_HHMMSS.txt`) - удобочитаемый формат

### Структура JSON

```json
{
  "search_date": "2024-01-15T10:30:00",
  "total_jobs": 20,
  "jobs": [
    {
      "title": "Python Developer Needed",
      "description": "...",
      "budget": "$100",
      "client_info": "client_name",
      "url": "https://...",
      "found_at": "2024-01-15T10:30:00",
      "relevance_score": 0.85
    }
  ]
}
```

## Примечания

⚠️ **Важно:** 
- Структура сайта Fiverr может меняться, что может повлиять на работу парсера
- Для стабильной работы рекомендуется использовать последние версии Chrome и ChromeDriver
- При использовании нейросети требуется больше оперативной памяти (минимум 8GB RAM для 3B модели)
- Уважайте условия использования Fiverr при автоматизированном доступе

## Лицензия

MIT License
