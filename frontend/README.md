# Backline Kitchen - Frontend

This Vite + React app calls the real Python backend API (`/api/...`) via `src/api/backendApi.js`.
Fixtures are fetched live from the official FPL feed and default to the current gameweek.

Run locally:

```bash
cd ..
python backend_api.py
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the printed localhost URL in your browser.
The Vite dev server proxies `/api` requests to `http://localhost:5000` (see `vite.config.js`).
