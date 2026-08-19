# UkrMediaNLP

Streamlit-додаток для збору новин з RSS українських медіа, скрейпінгу повного тексту та NLP-аналізу українською мовою.

Аналог проєкту [NLP_SyntA](https://github.com/prostir06/NLP_SyntA), адаптований для українських джерел і моделей.

Репозиторій: https://github.com/prostir06/UkrMediaNLP

## Можливості

- **4 категорії / 28 RSS-джерел:** Новини, Економіка, Спорт, Технології
- **SSRF-захист** при RSS і скрейпінгу (allowlist доменів, блок private IP)
- **Паралельний скрейпінг** (до 50 статей, 3 workers, rate limit 1 req/s) + SQLite TTL-кеш
- **NLP-аналіз українською:**
  - уніграми, біграми, триграми (з lemmatization)
  - ключові слова та хмара слів (контрастна палітра)
  - статистика тексту, NER, POS, LDA, сумаризація
  - тональність (новини) — завжди; RoBERTa / емоції — лише з `ALLOW_HEAVY_NLP=1`
  - порівняння медіа в межах категорії
- **Пошук у корпусі** — ключові слова/фрази або семантичний режим (`ALLOW_EMBEDDINGS=1`) по кількох медіа категорії
- **Тренди тем** — гібридні теми (авто + ручні) і порівняння медіа на графіку
- **Durable corpus (опційно)** — Postgres upsert / 90 днів retention / ingest CLI (`DATABASE_URL`)

## Структура проєкту

```
├── streamlit_app.py        # Entry point для Streamlit Cloud
├── runtime_env.py          # HF/torch defaults (ALLOW_HEAVY_NLP=0)
├── app.py                  # Thin Streamlit router (sidebar + dispatch)
├── ui/
│   ├── features/           # render_* screens (snapshot, ngrams, sentiment, corpus…)
│   ├── session_corpus.py   # load_source / corpus session helpers
│   ├── widgets.py          # shared Streamlit widgets
│   ├── corpus_controls.py  # corpus sidebar + multi-source load (+ optional store)
│   ├── corpus_charts.py    # corpus Plotly charts
│   ├── charts.py / renderers.py
├── corpus_store/           # Durable Postgres corpus (SQLAlchemy + ingest CLI)
├── alembic/                # DB migrations (articles)
├── cache.py                # Thin wrappers (моделі + load_articles → SQLite)
├── article_cache.py        # SQLite TTL-кеш статей
├── config.py               # NLP caps / function lists (re-exports media_sources)
├── media_sources.py        # NEWS_SOURCES TypedDict registry + sample URLs
├── data_loader.py          # fetch_articles (RSS + scrape)
├── observability.py        # structured step= / elapsed_ms logs
├── url_utils.py            # SSRF URL validation
├── rss.py                  # RSS-парсер (shared HTTP stack)
├── scraping.py             # HTTP-скрейпінг
├── scrapers/               # Site-specific + JSON-LD парсери
├── nlp/                    # NLP-модулі (без Streamlit)
├── data/stopwords_uk.txt   # Українські стоп-слова
├── requirements.txt        # Повний NLP (CPU torch) + Postgres driver
├── requirements-cloud.txt  # Light Cloud (без torch; sqlalchemy optional)
├── packages.txt            # fonts-dejavu-core для Streamlit Cloud
├── runtime.txt             # Python 3.12 для Streamlit Cloud
├── .env.example            # Local compose secrets template (not for prod)
├── tests/                  # pytest (~369 тестів)
├── Dockerfile
└── docker-compose.yml      # app + Postgres 16 (+ profiles; password from env)
```

## Встановлення (локально)

**Python 3.11+** (рекомендовано 3.12).

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
python -m spacy download uk_core_news_sm
```

Для розробки та тестів:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest -m "not slow" --cov=. --cov-fail-under=70
```

## Запуск локально

```bash
streamlit run streamlit_app.py
```

Або `streamlit run app.py`. Відкрийте http://localhost:8501

## Docker

```bash
# опційно: cp .env.example .env  (override local defaults)
docker compose up --build
```

Додаток: http://localhost:8501. Compose піднімає **Postgres 16** і передає `DATABASE_URL` у застосунок (durable corpus). Без `DATABASE_URL` у самому процесі Python локально — лише session-корпус; у Docker є дефолт на сервіс `postgres`.

**Secrets:** compose defaults / `.env.example` (`ukrmedia`) — лише local/CI. У production задайте сильний `POSTGRES_PASSWORD` і `DATABASE_URL`; не вважайте дефолти готовими до prod.

Міграції схеми:

```bash
# приклад для host → контейнер Postgres на :5432
set DATABASE_URL=postgresql+psycopg://ukrmedia:ukrmedia@localhost:5432/ukrmedia
alembic upgrade head
```

Пакетний ingest (після міграцій):

```bash
python -m corpus_store.ingest --all --dry-run
docker compose --profile ingest run --rm ingest
```

Preload моделей:

```bash
docker compose build --build-arg PRELOAD_MODELS=true
docker compose up
```

spaCy medium (`uk_core_news_md`):

```bash
docker compose --profile spacy-md up --build
```

Повний NLP у Docker (`ALLOW_HEAVY_NLP=1`, `mem_limit: 4g`):

```bash
docker compose --profile full-nlp up --build
```

Nginx reverse proxy:

```bash
docker compose --profile with-nginx up --build
```

> Моделі transformers (~1 GB) завантажуються при першому використанні тональності/емоцій. Volume `hf_cache` зберігає кеш між перезапусками.

### Streamlit Cloud (light)

1. Main file: `streamlit_app.py`
2. Requirements: `requirements-cloud.txt`
3. Опційно Secrets: `LIGHT_CLOUD = "1"` (читається з env і `st.secrets`)

Без torch доступні n-грами, NER, LDA, «Тональність (новини)» тощо.

## CI

GitHub Actions (`.github/workflows/tests.yml`):

- pytest coverage gate **≥ 70%** (omit: `app.py`, `streamlit_app.py`; `ui/*` включено)
- ruff lint
- cloud-deps job (`requirements-cloud.txt` + `LIGHT_CLOUD=1`, cov ≥50%)
- docker compose build smoke (+ Postgres + alembic)
- unit job Postgres service + `TEST_DATABASE_URL` dialect smoke

Weekly scraper health: `.github/workflows/scraper-health.yml`

## Деплой на Streamlit Community Cloud

Репозиторій: https://github.com/prostir06/UkrMediaNLP

### Рекомендовано (free tier / light)

1. [share.streamlit.io](https://share.streamlit.io) → **New app**
2. Repo `prostir06/UkrMediaNLP`, branch `main`
3. **Main file path:** `streamlit_app.py`
4. **Advanced → Python requirements file:** `requirements-cloud.txt`
5. Deploy (RoBERTa / емоції вимкнені за замовчуванням — стабільний режим)

`packages.txt` і `runtime.txt` (Python 3.12) підхоплюються автоматично.

Доступні: огляд, n-грами, keywords, wordcloud, NER, POS, LDA, «Тональність (новини)», сумаризація, порівняння медіа.

### Повний NLP (локально / Docker)

Потрібно ≥2 GB вільної RAM. Увімкніть важкі моделі явно:

```bash
# Windows PowerShell
$env:ALLOW_HEAVY_NLP="1"
streamlit run streamlit_app.py
```

Або Docker:

```bash
docker compose --profile full-nlp up --build
```

(легкий режим без важких моделей: `docker compose up --build`; за потреби `PRELOAD_MODELS=true`).

### Секрети (опційно)

```toml
# Стабільний Cloud (за замовчуванням і так light)
LIGHT_CLOUD = "1"

# Увімкнути RoBERTa / емоції (ризик OOM на free tier)
# ALLOW_HEAVY_NLP = "1"
```

## Підтримувані медіа

Категорії в сайдбарі (**Категорія → Медіа**). Повний реєстр — у `media_sources.NEWS_SOURCES` (також реекспорт з `config`).

| Категорія | К-сть | Приклади |
|-----------|------:|----------|
| Новини | 7 | NV, Liga.net, TSN, УНІАН, … |
| Економіка | 5 | Економічна правда, Бізнес Цензор, NV (Економіка), … |
| Спорт | 7 | Football.ua, Champion, Суспільне Спорт, Tribuna, … |
| Технології | 9 | DOU, AIN.UA, Speka, Mezha, ITC.ua, … |

## Конфігурація

| Параметр | За замовч. | Опис |
|----------|------------|------|
| `MAX_ARTICLES` | 50 | Макс. статей з RSS |
| `SCRAPE_MAX_WORKERS` | 3 | Паралельні потоки скрейпінгу |
| `MAX_SENTIMENT_TITLES` | 30 | Заголовків для тональності |
| `MAX_POS_ARTICLES` | 10 | Статей для POS-аналізу |
| `ARTICLE_CACHE_TTL` | 12h | TTL SQLite-кешу статей |
| `SPACY_MODEL` | `uk_core_news_sm` | spaCy pipeline |
| `ALLOW_HEAVY_NLP` | `0` | `1` = показати RoBERTa / емоції |
| `ALLOW_EMBEDDINGS` | `0` | `1` = semantic search mode in corpus UI (`nlp.embeddings`) |
| `DATABASE_URL` | _(немає)_ | Postgres URL для durable corpus; без нього — лише session |
| `POSTGRES_PASSWORD` | `ukrmedia` (compose) | Local default; override via `.env` — не для prod |

## Обмеження

- Sentiment-моделі натреновані переважно на соцмережах — на новинах точність може бути нижчою.
- Скрейпінг залежить від верстки сайтів.
- SSRF guard блокує fetch URL поза allowlist українських медіа.

## Ліцензія

MIT — див. [LICENSE](LICENSE).
