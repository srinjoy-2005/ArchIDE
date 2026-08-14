@echo off
setlocal

echo Starting Next.js frontend...
start "Next.js Frontend" cmd /c "npm run dev"

echo Starting FastAPI backend...
start "FastAPI Backend" cmd /c "cd backend && if exist .venv\Scripts\activate.bat (call .venv\Scripts\activate.bat) && python main.py"

echo Both servers are starting in separate windows.
echo   Frontend: http://localhost:3000
echo   Backend:  http://localhost:8001
echo Close the respective windows to stop the servers.
