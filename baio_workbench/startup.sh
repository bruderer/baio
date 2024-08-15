#!/bin/bash
echo "Starting Streamlit app"
echo "Python path:"
which python3
echo "Python version:"
python3 --version
echo "Pip list:"
pip3 list
echo "Sys path:"
python3 -c "import sys; print(sys.path)"
# Start the Streamlit app
streamlit run ./baio/baio_app.py
# open http://0.0.0.0:8501

