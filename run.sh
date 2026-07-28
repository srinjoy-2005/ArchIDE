#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Setup cleanup on exit to kill all background processes
cleanup() {
    echo "Stopping servers..."
    kill 0
}
trap cleanup SIGINT SIGTERM EXIT

echo "Starting Next.js frontend..."
npm run dev &

echo "Starting FastAPI backend..."
cd backend
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi
python main.py &

# Wait for all background jobs to finish
wait
