#!/bin/bash
# Stop and remove any running containers using the old image
docker ps -a | grep "report-evaluator" | awk '{print $1}' | xargs docker rm -f

# Rebuild fresh
docker build -t report-evaluator .

# Remove dangling images
docker image prune -f

# start container
docker run -v C:\Work\Bash\sp25_cs598DLH\cxr-baselines\evaluate:/evaluate/output -it --name my_container report-evaluator /bin/bash