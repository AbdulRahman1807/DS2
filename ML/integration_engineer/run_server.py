"""Simple Local Server Runner for Container Dwell Risk Predictor.

Run with:
    python3 ML/integration_engineer/run_server.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import uvicorn
from ML.integration_engineer.api import app


def main():
    print("=" * 65)
    print("  🚀 APM TERMINAL CONTAINER DWELL RISK PREDICTION SERVER")
    print("=" * 65)
    print("  • Local Dashboard: http://127.0.0.1:8000/dashboard")
    print("  • Root URL:        http://127.0.0.1:8000/")
    print("  • Interactive API: http://127.0.0.1:8000/docs")
    print("  • Health Check:    http://127.0.0.1:8000/health")
    print("=" * 65)
    print("  Model: XGBoost Bottleneck Classifier (Sub-25ms SLA)")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 65 + "\n")

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
