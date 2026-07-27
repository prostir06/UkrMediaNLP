# Design: Пошук у корпусі + Тренди тем

**Дата:** 2026-07-27  
**Проєкт:** UkrMediaNLP  
**Статус:** approved in brainstorming; awaiting user review of this file  

## Problem

Користувач аналізує одне медіа за раз. Немає спільного корпусу для кількох джерел однієї категорії, пошуку за ключовими словами та порівняння динаміки тем у часі між медіа.

## Goals

1. Два окремі пункти в «Функція»: **«Пошук у корпусі»** і **«Тренди тем»**.
2. Спільний завантажений корпус (shared corpus) для обох функцій.
3. Вибір медіа: multiselect у межах категорії + опція **«Вся категорія»**.
4. Гібридні теми: авто-пропозиції (lemmas / опційно LDA) + ручні терміни.
5. Графіки трендів і порівняння медіа (Plotly).
6. Орієнтація на **повний локальний NLP** (Cloud free tier не є критерієм успіху v1).

## Non-goals (v1)

- Embeddings / semantic search.
- Корпус крос-категорійний (медіа з різних категорій одночасно).
- Експорт PDF/HTML.
- Окрема SQLite-таблиця для корпусу (використовуємо існуючий per-source cache).
- Оптимізація під Streamlit Cloud light без scrape.

## Approach

**Shared corpus layer + два UI handlers** (обрано замість двох незалежних пайплайнів і замість важкого analytics/embeddings pipeline).

## Architecture

```
Sidebar: Категорія → (для двох функцій) corpus controls
         → Функція: «Пошук у корпусі» | «Тренди тем»

ui/corpus_controls.py     — multiselect, «вся категорія», дати, кнопка load
nlp/corpus.py             — load merge, date filter, search, trend aggregate, suggestions
ui/charts.py (або corpus charts) — Plotly line / bar helpers
app.py                    — render_corpus_search, render_topic_trends; session_state glue
config.py                 — NLP_FUNCTIONS_* + MAX_CORPUS_* константи
```

### Session state

| Key | Purpose |
|-----|---------|
| `corpus_category` | Категорія, з якою зібрано корпус |
| `corpus_sources` | `list[str]` обраних медіа |
| `corpus_df` | Об’єднаний DataFrame статей |
| `corpus_loaded_at` | Timestamp завантаження (інвалідація UX) |
| `corpus_date_from` / `corpus_date_to` | Період, застосований при load або post-filter |

Інвалідація: зміна категорії або набору джерел вимагає повторного натискання «Завантажити / оновити корпус». Не перезавантажувати RSS на кожний Streamlit rerun.

### Data loading

1. `sources_for_category(category)` → кандидати для multiselect.
2. «Вся категорія» → усі кандидати; інакше обраний subset (cap `MAX_CORPUS_SOURCES`, default 10).
3. Для кожного джерела: існуючий `load_articles(name, progress_callback)` (SQLite TTL уже всередині).
4. `pd.concat`; колонки як у `ARTICLE_COLUMNS` + `published_dt` (parsed).
5. Фільтр за `[date_from, date_to]`; checkbox «включити статті без дати».
6. Загальний cap `MAX_CORPUS_ARTICLES_TOTAL` (default 300): після concat обрізати за датою ↓.

Progress: `(done_sources, total_sources)` у sidebar.

Часткові збої: одне джерело впало → `warning` + продовжити; усі впали → `error`, не замінювати старий успішний `corpus_df`.

## UI

### Corpus controls (видимі лише для двох нових функцій)

- Multiselect медіа + checkbox «Вся категорія» (коли увімкнено — multiselect disabled).
- `date_input` від/до; після першого load — default min/max дат корпусу або останні 14 днів.
- Кнопка «Завантажити / оновити корпус» (явна, без auto-fetch).

### «Пошук у корпусі»

- Запит: слово або фраза.
- Де шукати: title / content(+description fallback) / обидва.
- Опції: «ціле слово»; «з урахуванням лем» (повільніше).
- Результат: таблиця (дата, медіа, заголовок-link, snippet ~120 символів); метрики N + bar по медіа (Plotly).
- Сортування: дата ↓, потім релевантність (входження в title важливіші за content).

### «Тренди тем»

- Авто: топ-10–15 lemmas з корпусу (`preprocess` + існуючі n-gram/keyword helpers).
- Expander «Поглиблено»: LDA-мітки (якщо ≥3 тексти); при помилці — warning, лише lemmas.
- Ручні теми: `text_area`, один термін на рядок.
- Multiselect обраних тем для графіка (max 8).
- Графік 1: Plotly line — X = день або тиждень (toggle), Y = кількість статей з match.
- Графік 2: порівняння медіа для **однієї** обраної теми (серії по `source`).

Match для трендів: case-insensitive підрядок у title+content; опційно lemma-match для однослівних тем.

## Algorithms (pure, testable)

### `parse_published` / `filter_by_date`

Нормалізація RSS дат у `published_dt`; фільтр інтервалу; опційний include_missing_dates.

### `search_corpus(df, query, fields, whole_word, use_lemmas) -> DataFrame`

Literal / whole-word regex (Unicode); snippet з контекстом.

### `suggest_terms(df, n) -> list[str]`

Top lemmas from titles (+ optional content sample).

### `aggregate_trends(df, terms, freq=['D'|'W-MON']) -> DataFrame`

Columns: `bucket`, `term`, `count` або для media compare: `bucket`, `source`, `count`.

### LDA suggestions

Reuse existing topic pipeline; soft-fail.

## Config

```text
MAX_CORPUS_SOURCES = 10
MAX_CORPUS_ARTICLES_TOTAL = 300
MAX_TREND_TERMS = 8
```

Додати «Пошук у корпусі» і «Тренди тем» до `NLP_FUNCTIONS_FULL` і до `NLP_FUNCTIONS_LIGHT`.  
У LIGHT: ті самі UI-пункти, але expander LDA прихований (лише lemma-пропозиції).

## Error handling

| Case | Behavior |
|------|----------|
| Partial source failure | warning, keep other sources |
| All sources fail | error, keep previous corpus if any |
| Empty after date filter | info: widen dates / include undated |
| LDA fail / too few docs | warning; lemma suggestions only |
| MemoryError | error: fewer sources / lower MAX_ARTICLES |
| No corpus loaded | info: press load button first |

## Testing

- `tests/test_corpus.py`: parse dates, filter, search (phrase/whole-word), trends D/W, multi-source groupby.
- `tests/test_corpus_suggest.py`: suggestions from mock DF.
- `tests/test_config.py`: new function names in `NLP_FUNCTIONS_FULL`.
- Optional: mock `load_articles` for corpus load wiring in app helpers.

No live network in unit tests.

## Success criteria

1. Користувач у категорії обирає 2+ медіа (або «вся категорія»), вантажить корпус один раз.
2. «Пошук у корпусі» знаходить статті за фразою з snippet і розбивкою по медіа.
3. «Тренди тем» показує лінії для кількох тем і другий графік порівняння медіа для однієї теми.
4. Авто-пропозиції + ручні терміни працюють разом.
5. Unit-тести покривають pure corpus логіку; ruff clean.

## Implementation order (hint for plan)

1. Config + `nlp/corpus.py` + unit tests.  
2. `ui/corpus_controls.py` + session_state load in `app.py`.  
3. Search UI + chart.  
4. Trends UI + dual charts + suggestions/LDA soft path.  
5. Wire NLP_FUNCTIONS + README note.
