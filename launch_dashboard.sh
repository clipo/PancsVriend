#!/bin/bash
# Launch the Schelling Segregation Dashboard

echo "🏘️ Launching Schelling Segregation Dashboard..."
echo "============================================"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "⚠️  Streamlit not found. Installing dashboard dependencies..."
    pip install -r requirements_dashboard.txt
fi

# Launch dashboard
echo "🚀 Starting dashboard at http://localhost:8501"
echo "Press Ctrl+C to stop"
streamlit run dashboard.py