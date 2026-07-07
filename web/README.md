# Llama-Emotion Web Chat

React (Vite) UI for the emotional chat API (`src/serve/microscope_api.py`).

## Local development

Terminal 1 — API (needs CUDA GPU + trained gate on disk or Modal volume mirror):

```powershell
py -3 -m pip install fastapi uvicorn
py -3 run_microscope.py
```

Terminal 2 — frontend:

```powershell
cd web
npm install
npm run dev
```

Open http://localhost:5173 (Vite proxies `/chat`, `/health`, etc. to port 8765).

## Production deploy (Modal — GPU + static UI)

```powershell
cd web
npm install
npm run build
cd ..
py -3 -m modal deploy run_web_serve.py
```

Modal prints a public HTTPS URL serving both the API and the built SPA.

## Environment

| Variable | Purpose |
|----------|---------|
| `VITE_API_URL` | API base URL when frontend is hosted separately (e.g. Vercel → Modal API) |
| `VITE_DEV_API` | Dev proxy target (default `http://127.0.0.1:8765`) |
| `CHAT_CORS_ORIGINS` | Comma-separated origins allowed by the API CORS middleware |
