# UkrMediaNLP



Streamlit-додаток для збору новин з RSS українських медіа, скрейпінгу повного тексту та NLP-аналізу українською мовою.



Аналог проєкту [NLP_SyntA](https://github.com/prostir06/NLP_SyntA), адаптований для українських джерел і моделей.



## Можливості



- **8 RSS-джерел:** NV, Радіо Свобода, Українська правда, Liga.net, RBC-UA, Інтерфакс-Україна, TSN, УНІАН

- **SSRF-захист** при скрейпінгу (allowlist доменів, блок private IP)

- **Паралельний скрейпінг** (до 50 статей, 3 workers, rate limit 1 req/s)

- **NLP-аналіз українською:**

  - уніграми, біграми, триграми (з lemmatization)

  - хмара слів

  - статистика тексту

  - NER (spaCy `uk_core_news_sm`)

  - частини мови

  - тематичне моделювання (LDA)

  - тональність (RoBERTa-COSMUS)

  - емоції (`ukr-emotions-classifier`)

  - екстрактивна сумаризація (LexRank)



## Структура проєкту



```

├── streamlit_app.py        # Entry point для Streamlit Cloud

├── app.py                  # Streamlit UI

├── cache.py                # Streamlit cache wrappers

├── config.py               # Джерела новин і константи

├── data_loader.py          # fetch_articles (RSS + scrape)

├── url_utils.py            # SSRF URL validation

├── rss.py                  # RSS-парсер

├── scraping.py             # HTTP-скрейпінг

├── scrapers/               # Site-specific + JSON-LD парсери

├── nlp/                    # NLP-модулі (без Streamlit)

├── nlp_analysis.py         # Фасад для UI

├── data/stopwords_uk.txt   # Українські стоп-слова

├── requirements.txt        # Python-залежності (CPU torch)

├── packages.txt            # Системні пакети для Streamlit Cloud

├── tests/                  # pytest (96+ тестів)

├── Dockerfile              # Multi-stage Docker image

└── docker-compose.yml

```



## Встановлення (локально)



**Python 3.11+** (рекомендовано 3.12).



```bash

python -m venv .venv

.venv\Scripts\activate        # Windows

# source .venv/bin/activate   # Linux/macOS



pip install -r requirements.txt

python -m spacy download uk_core_news_sm   # якщо модель не встановилась автоматично

```



Для розробки та тестів:



```bash

pip install -r requirements-dev.txt

pytest -m "not slow" --cov=. --cov-fail-under=65

```



## Запуск локально



```bash

streamlit run streamlit_app.py

```



Або `streamlit run app.py` — обидва entry points працюють однаково.



Відкрийте http://localhost:8501



## Docker



```bash

docker compose up --build

```



Додаток: http://localhost:8501



> Моделі transformers (~1 GB) завантажуються при першому використанні тональності/емоцій. Volume `hf_cache` зберігає кеш між перезапусками.



---



## Публікація на GitHub



### 1. Ініціалізація репозиторію



```bash

cd UkrMediaNLP

git init

git add .

git commit -m "Initial commit: UkrMediaNLP Streamlit app"

```



### 2. Створення репозиторію на GitHub



1. [github.com/new](https://github.com/new) → назва, напр. `UkrMediaNLP`

2. **Не** додавайте README/LICENSE (вони вже є локально)

3. Підключіть remote і запуште:



```bash

git branch -M main

git remote add origin https://github.com/YOUR_USERNAME/UkrMediaNLP.git

git push -u origin main

```



### 3. Що перевіряє CI



GitHub Actions (`.github/workflows/tests.yml`):



- pytest з coverage ≥ 65%

- ruff lint

- docker compose build (smoke)



Weekly scraper health check (`.github/workflows/scraper-health.yml`) — live RSS smoke.



### 4. Файли, які не потрапляють у git



- `.venv/`, `.pytest_cache/`, `.cache/`

- `.streamlit/secrets.toml` (секрети; для цього проєкту не потрібні)



---



## Деплой на Streamlit Community Cloud



### Передумови



- Публічний репозиторій на GitHub

- Обліковий запис [Streamlit Cloud](https://share.streamlit.io)



### Кроки



1. **New app** → підключити GitHub repo

2. **Branch:** `main`

3. **Main file path:** `streamlit_app.py`

4. **App URL:** обрати унікальний slug

5. **Deploy**



Streamlit Cloud автоматично використовує:



| Файл | Призначення |

|------|-------------|

| `requirements.txt` | Python-залежності (torch CPU, spaCy, transformers) |

| `packages.txt` | `fonts-dejavu-core` для хмари слів |

| `.streamlit/config.toml` | headless mode, без telemetry |



### Секрети



Цей проєкт **не потребує** API keys або `.streamlit/secrets.toml`. Усі RSS-джерела публічні.



### Обмеження Streamlit Cloud (free tier)



| Ресурс | Ліміт | Вплив |

|--------|-------|-------|

| RAM | ~1 GB | Тональність/емоції (transformers) можуть не вміститись |

| CPU | Shared | Перший scrape 50 статей ~15–30 с |

| Disk | Ephemeral | HF-моделі завантажуються при cold start |



**Рекомендації для Cloud:**



- Почніть з функцій без transformers: огляд статей, n-грами, NER, POS, LDA

- Тональність/емоції — для Docker/VPS (2 GB+ RAM) або локально

- При OOM Streamlit покаже помилку в UI; перезапустіть app або оберіть легшу функцію



### Альтернатива: Docker на VPS



Для production з усіма NLP-функціями:



```bash

docker compose up --build -d

# nginx reverse proxy + basic auth — опційно

```



---



## Підтримувані медіа



| Медіа | RSS |

|-------|-----|

| NV | https://nv.ua/ukr/rss/all.xml |

| Радіо Свобода | https://www.radiosvoboda.org/api/zrqitl-vomx-tpeoumq |

| Українська правда | https://www.pravda.com.ua/rss/view_mainnews/ |

| Liga.net | https://news.liga.net/ua/top/rss.xml |

| RBC-UA | https://www.rbc.ua/static/rss/ukrnet.strong.ukr.rss.xml |

| Інтерфакс-Україна | https://interfax.com.ua/news/last.rss |

| TSN | https://tsn.ua/rss/full.rss |

| УНІАН | https://rss.unian.ua/site/news_ukr.rss |



## Конфігурація



Ключові константи в `config.py`:



| Параметр | За замовч. | Опис |

|----------|------------|------|

| `MAX_ARTICLES` | 50 | Макс. статей з RSS |

| `SCRAPE_MAX_WORKERS` | 3 | Паралельні потоки скрейпінгу |

| `MAX_SENTIMENT_TITLES` | 30 | Заголовків для тональності |

| `MAX_POS_ARTICLES` | 10 | Статей для POS-аналізу |



## Обмеження



- Sentiment-моделі натреновані переважно на соцмережах — на новинах точність може бути нижчою.

- Скрейпінг залежить від верстки сайтів і може потребувати оновлення селекторів.

- SSRF guard блокує fetch URL поза allowlist українських медіа.



## Ліцензія



MIT — див. [LICENSE](LICENSE).


