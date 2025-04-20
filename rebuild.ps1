# Before running
# Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
# rebuild.ps1
Write-Host "Stopping and removing containers..." -ForegroundColor Cyan
docker ps -a | Select-String "report-evaluator" | ForEach-Object { 
    $id = $_.Line.Split()[0]
    docker rm -f $id
}

Write-Host "Building new image..." -ForegroundColor Cyan
docker build -t report-evaluator .

Write-Host "Cleaning up..." -ForegroundColor Cyan
docker image prune -f

Write-Host "Done!" -ForegroundColor Green

docker run -v C:\Work\Bash\sp25_cs598DLH\cxr-baselines\evaluate\output:/evaluate/output -it --name my_container report-evaluator /bin/bash
