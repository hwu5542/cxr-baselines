#!/bin/bash

# Check if commit message was provided
if [ -z "$1" ]; then
  echo "Error: Please provide a commit message as argument"
  echo "Usage: $0 \"Your commit message\""
  exit 1
fi

commitMessage="$1"

# Execute git commands with error checking
echo "Pulling latest changes..."
if ! git pull; then
  echo "Error: git pull failed"
  exit 1
fi

echo "Adding all changes..."
if ! git add .; then
  echo "Error: git add failed"
  exit 1
fi

echo "Committing with message: \"$commitMessage\""
if ! git commit -m "$commitMessage"; then
  echo "Error: git commit failed"
  exit 1
fi

echo "Pushing changes..."
if ! git push; then
  echo "Error: git push failed"
  exit 1
fi

echo "Successfully completed all git operations!"