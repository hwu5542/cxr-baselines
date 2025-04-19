#!/bin/bash
# Stop and remove any running containers using the old image
docker ps -a | grep "report-evaluator" | awk '{print $1}' | xargs docker rm -f

# Rebuild fresh
docker build -t report-evaluator .

# Remove dangling images
docker image prune -f