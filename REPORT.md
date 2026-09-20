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

- Docker Desktop изначально не был запущен: команды Docker не могли получить доступ к daemon.
  Диагностика — `docker info`; после запуска Desktop Compose и kind стали доступны.
- После установки `kind` и `k9s` новые команды не появились в уже открытом PowerShell из-за
  старого `PATH`. Решение — открыть новое окно PowerShell или терминал VS Code.
- В рабочем каталоге с кириллицей editable-установка через первоначальный backend создавала
  некорректный путь импорта. Проект переключён на `setuptools`; после этого `uv sync` и
  `uv run python -c "import credit_service"` проходят в том же каталоге.
- Сразу после `docker compose up -d --build` API мог ещё находиться в состоянии
  `health: starting`, из-за чего первый HTTP-запрос закрывался без ответа. Проверка
  `docker compose ps` и повторный запрос после перехода контейнера в `healthy` решили проблему.
- kind не обновляет образ с тем же тегом автоматически при `imagePullPolicy: IfNotPresent`.
  После новой сборки образ загружался через `kind load docker-image`, затем выполнялся
  `kubectl rollout restart deployment/credit-scoring-service`; ID образа дополнительно
  проверялся через `crictl images` внутри control-plane.
- Для Service выбран порт `80` с `targetPort: http` (`8000` в контейнере). Это позволило
  использовать одинаковую команду `kubectl port-forward service/credit-scoring-service 8080:80`
  при локальной проверке и в инструкции.
