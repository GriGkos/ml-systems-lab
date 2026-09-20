# Отчёт по домашней работе 1

## Модель

Сервис классифицирует вид ириса по четырём измерениям цветка: длине и ширине чашелистика,
длине и ширине лепестка. Артефакт содержит `sklearn.pipeline.Pipeline` с нормализацией и
классификатором, а также паспорт с версией, датой обучения, метрикой, хешем датасета,
зависимостями и примером корректного входа.

## Выполненные проверки

- `uv run pytest` — **9 passed**.
- `docker compose up -d --build` — `/health` и `/ready` вернули `200`.
- Compose: запрос с `request_id=compose-lecture-final-001` вернул `setosa`; строка появилась в
  `prediction_logs` с `model_version=1.0.0` и `status_code=200`.
- kind: Deployment `iris-service` развёрнут в двух репликах `Running`; Service использует
  `80 → 8000`; запрос через `kubectl port-forward service/iris-service 8080:80` с
  `request_id=k8s-lecture-final-001` вернул `setosa`.

## Скриншоты перед сдачей

Нужно вручную добавить реальные скриншоты фактического терминала и k9s:

1. `uv run pytest` с зелёным результатом;
2. `SELECT` из `prediction_logs` после запроса к API;
3. `kubectl get pods` с двумя репликами и ответ `/v1/predict` через port-forward;
4. экран `k9s` с подами сервиса.

## Журнал проблем

- Docker Desktop сначала не был запущен; после запуска `docker info` и `kind create cluster`
  отработали успешно.
- После установки `kind` и `k9s` PATH обновился только в новом окне PowerShell/терминала VS Code.
- Локальный путь содержит кириллицу, поэтому для editable-установки выбран `setuptools`; команда
  `uv sync` и импорт пакета проходят корректно.
