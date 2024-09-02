# BaIO Streamlit App Documentation

## File: `baio/baio_app.py`

### Overview

This file contains the main Streamlit application for BaIO. It sets up the user interface and coordinates the different components of the system.

### Key Components:

1. **Imports and Setup**:
   - Imports necessary libraries and modules
   - Sets up file paths for various resources

2. **Helper Functions**:
   - `read_txt_file()`: Reads content from text files

3. **Main Application Function** (`app()`):
   - Sets up the Streamlit interface
   - Handles user authentication (OpenAI API key input)
   - Initializes the language model and embedding
   - Manages agent selection and execution

### User Interface Elements:

1. **Sidebar**:
   - OpenAI API key input
   - Model selection dropdown
   - "Reinitialize LLM" button
   - Information about BaIO

2. **Main Area**:
   - Agent selection radio buttons
   - Query input areas
   - File upload functionality
   - Results display

### Agents and Tools:

The app integrates several agents:
- ANISEED Agent
- BaIO Agent (for NCBI queries)
- Local GO Agent
- Local File Agent (CSV Chatter)

Each agent has its own section in the UI with specific instructions and input fields.

### File Management:

- Utilizes the `FileManager` class to handle file uploads, downloads, and previews
- Supports various file types including CSV, TXT, and JSON

### Error Handling and Callbacks:

- Implements try-except blocks for error handling
- Uses OpenAI callbacks to track API usage and costs

### Conditional Rendering:

- The full functionality is only available when a valid OpenAI API key is provided
- Without an API key, limited functionality is displayed along with general information about BaIO

### Usage Instructions:

1. Run the Streamlit app: `streamlit run baio/baio_app.py`
2. Input your OpenAI API key in the sidebar
3. Select the desired agent/tool
4. Enter your query or upload files as needed
5. View results and download files as required

This Streamlit app serves as the user-friendly interface for BaIO, making it easy for biologists and bioinformaticians to leverage the power of natural language processing for biological data analysis.