# DomainSorter

## Как запустить
### Клонирование репозитория
```bash
git clone https://github.com/bestaicoder/DomainSorter.git
```

### Предварительные требования
1. Установите переменные окружения:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   export JINA_API_TOKEN="your_jina_api_token"
   ```

2. Установите зависимости:
   ```bash
   pip3 install -r requirements.txt
   ```

### Шаги выполнения

1. **Загрузка данных доменов** (требует `Input/domains.txt` и `Input/categories.txt`):
   ```bash
   python3 DownLDom.py
   ```
   → Генерирует `Results/ResultsData.json`

2. **Вычисление схожести категорий** (требует `Input/categories.txt` и `Results/ResultsData.json`):
   ```bash
   python3 CatSim.py
   ```
   → Генерирует `Results/ratings.json`

3. **Проверка категоризации с помощью LLM** (требует `Results/ratings.json` и `Results/ResultsData.json`):
   ```bash
   python3 LlmVerif.py
   ```
   → Генерирует `Results/ResultsVerif.json`

### Необходимые входные файлы
- `Input/domains.txt` - Список доменов для обработки
- `Input/categories.txt` - Список категорий, разделенных точкой с запятой

### Генерируемые выходные файлы
- `Results/ResultsData.json` - Сырые данные доменов из Jina API
- `Results/ratings.json` - Рейтинги схожести категорий для каждого домена
- `Results/ResultsVerif.json` - Категоризации, проверенные LLM
