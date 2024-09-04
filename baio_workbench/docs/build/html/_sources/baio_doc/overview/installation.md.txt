
## Installation 

BaIO can be easily deployed using either Docker Compose or Poetry:

### Docker Compose

1. Ensure Docker and Docker Compose are installed on your system.
2. Clone the BaIO repository.
   ```
   wget https://github.com/bruderer/baio
   ```

3. Navigate to the project directory.
4. Run:
   ```
   docker-compose up -d
   ```
5. Access the app at `http://localhost:8501` in your web browser.

### Poetry

1. Ensure Python 3.10+ and Poetry are installed on your system.
2. Clone the BaIO repository.
3. Navigate to the project directory.
4. Install dependencies:
   ```
   poetry install
   ```
5. Run the Streamlit app:
   ```
   poetry run streamlit run baio/baio_app.py
   ```
6. Access the app at the URL provided in the terminal.

With these simple deployment options, you can quickly set up BaIO and start leveraging its powerful features for your biological data analysis needs.