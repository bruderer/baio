        
import os
import pandas as pd
import streamlit as st
import base64
import io
import contextlib
import sys
import threading
import time

from src.agents import ncbi_agent

def preview_file(file_path):
    """Preview the content of the selected file based on its type."""
    filename = os.path.basename(file_path)
    file_extension = os.path.splitext(filename)[1].lower()

    # Get file size in MB
    file_size = os.path.getsize(file_path) / (1024 * 1024)

    # Preview for image files
    if file_extension in ['.png', '.jpg', '.jpeg']:
        st.image(file_path)

    # Preview for text files
    elif file_extension in ['.txt', '.md', '.log']:
        with open(file_path, 'r') as f:
            st.text(f.read())

    # Preview for CSV files
    elif file_extension == '.csv':
        if file_size > 50:  # If file size is greater than 50 MB, read only first few rows
            df = pd.read_csv(file_path, nrows=1000)
        else:
            df = pd.read_csv(file_path)
        st.write(df)

    # Add more preview cases for other file types as needed

    else:
        st.warning(f"No preview available for {filename}.")

def file_download_button(path, label="Download"):
    """Generate a button to download the file."""
    file_path = path
    filename = os.path.basename(file_path)  # extract the filename
    with open(file_path, "rb") as f:
        bytes_data = f.read()
    b64 = base64.b64encode(bytes_data).decode()  # bytes to base64 string
    href = f'<a href="data:file/octet-stream;base64,{b64}" download="{filename}" style="display:inline-block;padding:0.25em 0.5em;background:#4CAF50;color:white;border-radius:3px;text-decoration:none">{label}</a>'
    return href


def save_uploaded_file(uploaded_file, UPLOAD_DIR):
    """Save the uploaded file to the specified directory."""
    with open(os.path.join(UPLOAD_DIR, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getvalue())

@contextlib.contextmanager
def capture_output():
    new_out = io.StringIO()
    old_out = sys.stdout
    try:
        sys.stdout = new_out
        yield sys.stdout
    finally:
        sys.stdout = old_out
# Custom output stream
class OutputCapture:
    def __init__(self):
        self.buffer = io.StringIO()

    def write(self, message):
        self.buffer.write(message)

    def flush(self):
        pass  # Not needed for StringIO

    def get_value(self):
        return self.buffer.getvalue()

# Function to run ncbi_agent in a thread
def run_ncbi_agent(question, output_queue, callback_info):
    try:
        result = ncbi_agent(question)
        output_queue.put(result)
    except Exception as e:
        output_queue.put(str(e))
    finally:
        callback_info['cost'] = callback_info.get('total_cost', 'Unknown')

