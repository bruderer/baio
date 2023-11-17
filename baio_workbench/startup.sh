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
echo "Trying to import mygene:"
python3 -c "import mygene; print('mygene imported successfully')"
# Start the Streamlit app
streamlit run /usr/src/app/baio/baio_app.py
