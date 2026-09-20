$ErrorActionPreference = 'Stop'

$image = 'credit-scoring-service:2.0'
$metadata = Get-Content -Raw artifacts/credit_scoring_metadata.json | ConvertFrom-Json
$payload = @{ features = $metadata.example_input } | ConvertTo-Json -Depth 4 -Compress

docker build -t $image .
kind load docker-image $image --name kind
kubectl apply -f k8s
kubectl rollout restart deployment/credit-scoring-service
kubectl rollout status deployment/credit-scoring-service --timeout=180s
kubectl get pods -l app=credit-scoring-service

$portForward = Start-Job -ScriptBlock {
    kubectl port-forward service/credit-scoring-service 8080:80
}

try {
    Start-Sleep -Seconds 3
    Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8080/v1/predict' -ContentType 'application/json' -Body $payload
}
finally {
    Stop-Job $portForward -ErrorAction SilentlyContinue
    Remove-Job $portForward -Force -ErrorAction SilentlyContinue
}
