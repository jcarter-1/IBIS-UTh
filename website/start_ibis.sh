#!/bin/bash

# ==================================================
# IBIS — Bayesian U–Th Age Estimation
# Local Website Launcher
# ==================================================

set -e pipefail

# Navigate to the folder containing this script
cd "$(dirname "$0")"

echo ""
echo "======================================"
echo "  IBIS —  
		A  medium-to-large, 
	    long-legged wading bird 
	    of the family 
	    Threskiornithidae,
		known for its slender, 
		downward-curving bill"
echo "======================================"
echo ""

if [ ! -x ".venv/bin/python" ]; then
	
	# Use the project's virtual environment when available
	if command -v python3.14>/dev/null 2>&1; then
	        PYTHON="python3.12"
	elif command -v python3.14 >/dev/null 2>&1; then
	    PYTHON="python3"
    else
        echo "ERROR: Python 3 is not installed."
        echo "Please install Python and try again."
        exit 1
    fi
    echo "Creating Python environment..."
    "$PYTHON" -m venv .venv
fi

PYTHON=".venv/bin/python"

# ------------------------------------------
# 2. Install dependencies if needed
# ------------------------------------------

if [ ! -f ".venv/.installed" ] || \
   [ "requirements.txt" -nt ".venv/.installed" ]; then

    echo "Installing Python dependencies..."
    "$PYTHON" -m pip install -r requirements.txt

    touch .venv/.installed

    echo "Dependencies installed successfully."
fi

# Local website address
URL="http://127.0.0.1:8000"

echo "Starting IBIS locally..."
echo "Website: $URL"
echo ""
echo "Press Ctrl+C to stop the website."
echo ""

# Open the browser after the server starts
(
    sleep 2
    if command -v open >/dev/null 2>&1; then
        open "$URL"
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$URL"
    fi
) &

# Start the FastAPI website
exec "$PYTHON" -m uvicorn app:app \
    --host 127.0.0.1 \
    --port 8000 \
    --no-access-log \
    --log-level warning