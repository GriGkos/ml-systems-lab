# Отчёт по домашней работе 1

## Модель

Сервис оценивает вероятность дефолта по анкете Home Credit Default Risk. Артефакт — единая
`sklearn` Pipeline с детерминированными признаками заявки, заполнением пропусков, one-hot
encoding, масштабированием и Logistic Regression. Модель обучена на 307 511 заявках без подбора
гиперпараметров.

Полная CatBoost v2 из исходного проекта требует CUDA и всех исторических таблиц, поэтому для
сервиса выбран воспроизводимый application-only baseline с исходными фиксированными параметрами.
Reference OOF ROC-AUC из проекта — `0.75141`.

## Выполненные проверки

- `uv run pytest` — **9 passed**.
- Compose: запрос с `request_id=credit-compose-renamed-001` вернул `prediction=default`,
  `default_probability=0.937867`; строка появилась в `prediction_logs` с `model_version=2.0.0`
  и `status_code=200`.
- kind: Deployment `credit-scoring-service` развёрнут в двух репликах `Running`; Service
  использует `80 → 8000`; запрос через
  `kubectl port-forward service/credit-scoring-service 8080:80` с
  `request_id=credit-k8s-renamed-001` вернул `prediction=default`.

## Скриншоты перед сдачей

Нужно вручную добавить реальные скриншоты фактического терминала и k9s:

1. `uv run pytest` с зелёным результатом;
2. `SELECT` из `prediction_logs` после запроса к API;
3. `kubectl get pods` с двумя репликами и ответ `/v1/predict` через port-forward;
4. экран `k9s` с подами сервиса.

## Журнал проблем

- Полная CatBoost v2 не обучалась: локально нет CUDA, а исходная версия требует исторические CSV
  общим объёмом около 2.5 ГБ и рассчитана на GPU.
- Фиксированный `saga`-solver дошёл до `max_iter=300` с предупреждением о недосходимости;
  параметры сознательно не менялись, чтобы не подбирать гиперпараметры.
