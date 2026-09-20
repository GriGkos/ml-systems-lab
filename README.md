# Iris classification service

FastAPI-сервис предсказывает вид ириса по четырём числовым признакам. Модель обучена на
открытом датасете Iris из `scikit-learn`; это самостоятельная модель, не churn-сервис с
семинара.

## Проверка

Выполняйте команды сверху вниз из корня репозитория.

```powershell
uv run pytest
```

```powershell
docker compose up -d --build
```

```powershell
.\scripts\verify_kind.ps1
```

Третья команда собирает образ, загружает его в текущий kind-кластер, применяет манифесты,
ожидает rollout и получает ответ `/v1/predict` через port-forward.

## API

После `docker compose up -d --build` доступны:

- `GET http://127.0.0.1:8000/health`
- `GET http://127.0.0.1:8000/ready`
- `POST http://127.0.0.1:8000/v1/predict`
- `GET http://127.0.0.1:8000/docs`

Пример запроса:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/predict -ContentType 'application/json' -Body '{"sepal_length_cm":5.1,"sepal_width_cm":3.5,"petal_length_cm":1.4,"petal_width_cm":0.2}'
```

Проверить логи в PostgreSQL:

```powershell
docker compose exec postgres psql -U iris -d iris -c "SELECT request_id, model_version, prediction, latency_ms, status_code FROM prediction_logs;"
```

## Локальная разработка

```powershell
uv sync
uv run python scripts/train_model.py
uv run uvicorn iris_service.main:app --reload
```

Переменная `DATABASE_URL` необязательна. Без неё сервис отвечает на запросы, но не пишет
логи в PostgreSQL.

## Структура

- `artifacts/` — joblib-бандл Pipeline и паспорт модели.
- `src/iris_service/` — исходный код FastAPI-сервиса.
- `tests/` — контрактные, smoke и детерминированные тесты.
- `k8s/` — Deployment и Service для kind.
- `REPORT.md` — журнал запуска и сдачи.
