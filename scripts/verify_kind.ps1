$ErrorActionPreference = 'Stop'

$image = 'iris-service:1.0'
$payload = '{"sepal_length_cm":5.1,"sepal_width_cm":3.5,"petal_length_cm":1.4,"petal_width_cm":0.2}'

docker build -t $image .
kind load docker-image $image --name kind
kubectl apply -f k8s
kubectl rollout restart deployment/iris-service
kubectl rollout status deployment/iris-service --timeout=180s
kubectl get pods -l app=iris-service

$portForward = Start-Job -ScriptBlock {
    kubectl port-forward service/iris-service 8080:8000
}

try {
    Start-Sleep -Seconds 3
    Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8080/v1/predict' -ContentType 'application/json' -Body $payload
}
finally {
    Stop-Job $portForward -ErrorAction SilentlyContinue
    Remove-Job $portForward -Force -ErrorAction SilentlyContinue
}
