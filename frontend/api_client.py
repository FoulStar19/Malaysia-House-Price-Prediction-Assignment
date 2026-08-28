"""Thin wrapper around requests calls to the backend API.

All functions are cached with Streamlit's cache decorators where sensible so
the app doesn't hammer the backend on every rerun (Streamlit reruns the
whole script on almost every interaction).
"""
import os

import requests
import streamlit as st

def _get_backend_url() -> str:
    """Set BACKEND_URL as a Streamlit secret (.streamlit/secrets.toml) or env
    var when deploying. Falls back to localhost for local development.
    st.secrets raises (rather than returning a default) when no
    secrets.toml exists at all, so this has to be defensive.
    """
    try:
        if "BACKEND_URL" in st.secrets:
            return st.secrets["BACKEND_URL"]
    except Exception:
        pass
    return os.environ.get("BACKEND_URL", "http://localhost:8000")


BACKEND_URL = _get_backend_url()
TIMEOUT = 60  # generous: free-tier backend hosts (Render/Railway/Fly) sleep
              # when idle and can take 30-50s to wake up on first request


class BackendError(RuntimeError):
    pass


def _get(path: str, params: dict | None = None):
    try:
        r = requests.get(f"{BACKEND_URL}{path}", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        raise BackendError(f"Couldn't reach the backend at {BACKEND_URL}{path}: {e}") from e


def _post(path: str, json_body: dict):
    try:
        r = requests.post(f"{BACKEND_URL}{path}", json=json_body, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        raise BackendError(f"Couldn't reach the backend at {BACKEND_URL}{path}: {e}") from e


@st.cache_data(ttl=300)
def get_meta():
    return _get("/meta")


@st.cache_data(ttl=60)
def get_listings(states=None, property_types=None):
    params = {}
    if states:
        params["state"] = states
    if property_types:
        params["property_type"] = property_types
    return _get("/listings", params=params)


@st.cache_data(ttl=300)
def get_listing(index: int):
    return _get(f"/listings/{index}")


@st.cache_data(ttl=60)
def get_state_summary():
    return _get("/market/state-summary")


@st.cache_data(ttl=300)
def get_model_comparison():
    return _get("/model/comparison")


@st.cache_data(ttl=300)
def get_model_diagnostics():
    return _get("/model/diagnostics")


def predict(payload: dict):
    return _post("/predict", payload)
