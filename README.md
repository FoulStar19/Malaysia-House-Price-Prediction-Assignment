# Malaysia Condo Price Predictor — Backend + Frontend

This splits the original single-file Streamlit app into two independent
services:

```
project/
├── backend/    FastAPI service — owns the model, scaler, and all artifacts
└── frontend/   Streamlit app — talks to the backend over HTTP, adds a
                map-based state picker (pydeck)
```

The frontend has **no model-loading code at all** anymore — it only calls
the backend's REST API and renders the response. That's what makes a
richer UI (like the map) possible: the frontend is now free to be whatever
you want, and the backend can be swapped, scaled, or even rewritten in a
different language without touching it.

## What each service does

**Backend (`backend/`)**
- Loads `best_model.pkl`, `scaler.pkl`, `feature_columns.pkl`,
  `all_results.pkl`, `extra_artifacts.pkl`, `app_sample_listings.csv` once
  at startup.
- Exposes:
  - `GET /health` — liveness check
  - `GET /meta` — dropdown options + state map coordinates
  - `GET /listings` — sample listings, filterable by state/property type
  - `GET /listings/{index}` — single listing (used for autofill)
  - `GET /market/state-summary` — avg price + count per state (feeds the map)
  - `GET /model/comparison` — RMSE/MAE/R² table + best model
  - `GET /model/diagnostics` — residuals, bracket confusion matrix, etc.
  - `POST /predict` — takes the form payload, returns prediction + context

**Frontend (`frontend/`)**
- Same 3 tabs as before (Market Overview / Model Comparison / Price
  Predictor), but all data comes from the backend via `api_client.py`.
- The state picker in the Predictor tab is now a **pydeck map**: bubble
  size = number of sample listings in that state, colour = average price
  (blue = cheaper, red = pricier). Click a bubble to select that state, or
  use the dropdown next to it — both stay in sync.

## Running locally

```bash
# Terminal 1 — backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

The frontend defaults to `http://localhost:8000` for the backend — no
config needed locally.

## Deploying for free

Streamlit Cloud only runs one Python process per app, so the backend needs
to live somewhere else. Any of these free tiers work well for a small
FastAPI service:

- **Render** (free web service) — easiest, has a native Python/uvicorn
  buildpack, this repo's `backend/Procfile` works as-is.
- **Railway** (free trial credits, then usage-based)
- **Fly.io** (free allowance)
- **Hugging Face Spaces** (Docker SDK, free CPU)

### 1. Deploy the backend

Using Render as the example:
1. Push `backend/` to a GitHub repo (or push the whole `project/` and set
   Render's "root directory" to `backend`).
2. Create a new **Web Service** on Render, point it at the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT` (already
   in the `Procfile`, Render picks it up automatically).
5. Deploy. Note the public URL, e.g. `https://your-app.onrender.com`.

> `best_model.pkl` is ~70MB. That's fine for Git and for Render's free
> tier disk, but if you use GitHub you may want Git LFS if the file grows
> much larger.

> Free tiers on Render/Railway/Fly spin the service down after inactivity.
> The first prediction after a period of idleness can take 30-50 seconds
> while it wakes up — the frontend's request timeout is already set high
> enough (60s) to ride that out, just expect a pause on the first request.

### 2. Deploy the frontend

1. Push `frontend/` to Streamlit Cloud (or the same repo, root dir
   `frontend`).
2. In the app's **Settings → Secrets**, add:
   ```toml
   BACKEND_URL = "https://your-app.onrender.com"
   ```
3. Deploy. The app reads `BACKEND_URL` from secrets automatically
   (`api_client.py` falls back to `localhost:8000` only when no secret or
   env var is set).

That's it — two independently deployed, independently scalable services.

## Extending the UI further

Because the frontend only talks to a JSON API now, you're not limited to
Streamlit for future UI work. The same backend could serve a React app, a
Next.js site, or a mobile app without any changes — `POST /predict` with
the same payload shape works from anywhere.
