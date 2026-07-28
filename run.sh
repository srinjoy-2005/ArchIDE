#!/bin/bash

# Store child PIDs so we can kill them precisely
PIDS=()

cleanup() {
    echo ""
    echo "Stopping servers..."

    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null
        fi
    done

    # Give processes a moment to shut down gracefully
    sleep 1

    # Force-kill anything still running
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null
        fi
    done

    echo "All servers stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "Starting Next.js frontend..."
npm run dev &
PIDS+=($!)

echo "Starting FastAPI backend..."
(
    cd backend
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    fi
    python main.py
) &
PIDS+=($!)

echo "Both servers running. Press Ctrl+C to stop."
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8001"
echo ""

# Wait for all background jobs to finish
wait
