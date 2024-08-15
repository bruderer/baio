import os
from pathlib import Path

import streamlit as st
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from baio.src.llm import LLM
from baio.st_app import MyAgents
from baio.st_app.file_manager import FileManager
from baio.st_app.helper_functions import save_uploaded_file

BASE_DIR = Path("./baio")
DATA_DIR = BASE_DIR / "data"
ST_APP_DIR = BASE_DIR / "st_app"
TEXT_CONTENT_DIR = ST_APP_DIR / "text_content"

# Define paths
UPLOAD_DIR = DATA_DIR / "uploaded"
DOWNLOAD_DIR = DATA_DIR / "output"

side_bar_txt_path = TEXT_CONTENT_DIR / "side_bar_text.txt"
aniseed_instruction_txt_path = TEXT_CONTENT_DIR / "aniseed_agent_instructions.txt"
go_file_annotator_instruction_txt_path = (
    TEXT_CONTENT_DIR / "go_annotator_instructions.txt"
)
csv_instruction_txt_path = TEXT_CONTENT_DIR / "csv_chatter_instructions.txt"
ncbi_instruction_txt_path = TEXT_CONTENT_DIR / "ncbi_agent_instructions.txt"
LICENSE_path = TEXT_CONTENT_DIR / "LICENSE.txt"


# initialising paths
base_dir = Path(
    os.getcwd()
)  # Gets the current working directory from where the app is launched

# Define paths using pathlib
path_aniseed_out = base_dir / "baio" / "data" / "output" / "aniseed" / "aniseed_out.csv"
path_go_nl_out = (
    base_dir / "baio" / "data" / "output" / "gene_ontology" / "go_annotation.csv"
)
go_file_out = (
    base_dir / "baio" / "data" / "output" / "gene_ontology" / "go_annotation.csv"
)


# Function to read text file using pathlib
def read_txt_file(path):
    path = Path(path)  # Ensure the path is a pathlib.Path object
    with path.open("r") as file:  # Use pathlib's open method
        return file.read()


def app():
    st.sidebar.markdown("""PROVIDE AN OpenAI API KEY:""")

    banner_image = "./baio/data/persistant_files/baio_logo.png"
    st.image(banner_image, use_column_width=True)
    openai_api_key = st.sidebar.text_input("OpenAI API KEY")
    model_options = ["gpt-4o", "gpt-4", "gpt-4-32k", "gpt-3.5-turbo"]
    default_model = "gpt-4o"
    selected_model = st.sidebar.selectbox(
        "Select a model",
        model_options,
        index=model_options.index(default_model),  # Set the default option by index
    )

    # Check if the "Reinitialize LLM" button is clicked or if the llm is not in session
    # state
    if st.sidebar.button("Reinitialize LLM") or "llm" not in st.session_state:
        if openai_api_key:
            LLM.initialize(openai_api_key=openai_api_key, selected_model=selected_model)
            st.sidebar.success(
                "LLM reinitialized with selected model!"
            )  # Show success message in the sidebar
        else:
            st.sidebar.error(
                "Please provide an OpenAI API key."
            )  # Show error message in the sidebar
    st.sidebar.markdown(read_txt_file(side_bar_txt_path), unsafe_allow_html=True)

    if st.sidebar.button("Show License"):
        st.sidebar.markdown(read_txt_file(LICENSE_path))
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=openai_api_key)
        embedding = OpenAIEmbeddings(api_key=openai_api_key)
        agent = MyAgents()
        selected_agent = agent.initialise_agent_selection()

        from baio.src.agents import aniseed_agent, baio_agent, csv_chatter_agent

        # ANISEED AGENT

        if selected_agent == agent.aniseed_agent:
            file_manager_aniseed = FileManager()
            with st.form("form_for_aniseed_agent"):
                st.write("Aniseed agent")
                with st.expander("Instructions"):
                    st.markdown(read_txt_file(aniseed_instruction_txt_path))
                question = st.text_area(
                    "Enter text for ANISEED agent:",
                    "Example: What genes are expressed between stage 1 and 3 in ciona"
                    " robusta?",
                )
                submitted = st.form_submit_button("Submit")
                if submitted:
                    with get_openai_callback() as cb:
                        try:
                            print("try")
                            result = aniseed_agent(question, llm)
                            # st.info(result['output'])
                            st.info(f"Total cost is: {cb.total_cost} USD")
                            st.write("Files generated:\n" + "\n".join(result))
                            file_manager_aniseed.preview_file(result[0])
                            st.markdown(
                                file_manager_aniseed.file_download_button(
                                    path_aniseed_out
                                ),
                                unsafe_allow_html=True,
                            )

                        except:
                            st.write(
                                "Something went wrong, please try to reformulate your "
                                "question"
                            )
                # if reset_memory:
                #     aniseed_go_agent.memory.clear()
            file_manager = FileManager(UPLOAD_DIR, DOWNLOAD_DIR)
            file_manager.run()

        # FILE GO ANNOTATOR
        elif selected_agent == agent.go_file_annotator:
            from baio.src.non_llm_tools import go_file_tool

            go_file_annotator_file_manager = FileManager(UPLOAD_DIR, DOWNLOAD_DIR)
            st.write("Local GO agent")
            with st.expander("Instructions"):
                st.markdown(read_txt_file(go_file_annotator_instruction_txt_path))
            uploaded_file = st.file_uploader("Upload your file below:", type=["csv"])
            if uploaded_file:
                st.write("You've uploaded a file!")
                save_uploaded_file(uploaded_file, UPLOAD_DIR)
                go_file_annotator_file_manager = FileManager(UPLOAD_DIR, DOWNLOAD_DIR)

            file_path = go_file_annotator_file_manager.select_file_preview_true(
                key="file_path"
            )

            with st.form("form_for_file_annotator_agent"):

                input_file_gene_name_column = st.text_area(
                    "Enter gene name column:", "gene_name"
                )
                submitted = st.form_submit_button("Submit")

                if submitted:
                    with get_openai_callback() as cb:

                        result = go_file_tool(file_path, input_file_gene_name_column)
                        go_file_annotator_file_manager = FileManager(
                            UPLOAD_DIR, DOWNLOAD_DIR
                        )

                        try:
                            go_file_annotator_file_manager.preview_file(result[1])
                            st.markdown(
                                go_file_annotator_file_manager.file_download_button(
                                    result[1]
                                ),
                                unsafe_allow_html=True,
                            )
                            st.info(cb.total_cost)
                        except TypeError:
                            st.info(result)

            file_manager = FileManager("./baio/data/output/", "./baio/data/upload/")
            file_manager.run()

        # FILE CHATTER

        # CHAT WITH YOUR CSV AND OTHER FILES
        elif selected_agent == agent.file_chatter:
            csv_chatter_file_manager = FileManager(UPLOAD_DIR, DOWNLOAD_DIR)
            st.write("Ask questions about your selected or uploaded file")
            with st.expander("Instructions"):
                st.markdown(read_txt_file(csv_instruction_txt_path))
            uploaded_file = st.file_uploader(
                "Upload your file with genes here", type=["csv", "xlsx", "txt"]
            )
            if uploaded_file:
                st.write("You've uploaded a file!")
                save_uploaded_file(uploaded_file, UPLOAD_DIR)
                csv_chatter_file_manager = FileManager(UPLOAD_DIR, DOWNLOAD_DIR)
            st.write("Select file 1:")
            file_path1 = csv_chatter_file_manager.select_file_preview_false(
                key="select_file_1"
            )
            st.write("Select file 2:")
            file_path2 = csv_chatter_file_manager.select_file_preview_false(
                key="select_file_2"
            )

            with st.form("test form"):
                st.write("Explore a file ")
                question = st.text_area(
                    "Enter text:",
                    "Example: What are the unique genes per stage? please make a new "
                    "data frame of them and put it in a file",
                )

                submitted = st.form_submit_button("Submit")

                if submitted:
                    files = [file_path1, file_path2]

                    with get_openai_callback() as cb:

                        if len(files) != 0:
                            result = csv_chatter_agent(question, files, llm)

                        try:
                            st.info(result)
                            st.info(cb.total_cost)
                        except TypeError:
                            st.info(result)

            file_manager = FileManager("./baio/data/output/", "./baio/data/upload/")
            file_manager.run()

        # NCBI

        elif selected_agent == agent.baio_agent:

            file_manager_aniseed = FileManager()
            with st.form("form_for_aniseed_agent"):
                st.write("NCBI agent")
                with st.expander("Instructions"):
                    st.markdown(read_txt_file(ncbi_instruction_txt_path))
                question = st.text_area(
                    "Enter question for NCBI agent:",
                    "Which organism does the DNA sequence come from:AGGGGCAGCAAACACCGGG"
                    "ACACACCCATTCGTGCACTAATCAGAAACTTTTTTTTCTCAAATAATTCAAACAATCAAAATTGGT"
                    "TTTTTCGAGCAAGGTGGGAAATTTTTCGAT",
                )
                submitted = st.form_submit_button("Submit")
                # st.write(path_aniseed_out, path_go_nl_out)
                # Add the file paths to the file_paths dictionary

                if submitted:
                    with get_openai_callback() as cb:
                        try:
                            result = baio_agent(question, llm, embedding)
                            st.info(result)
                            st.info(f"Total cost is: {cb.total_cost} USD")
                            st.write("Your generated file is below:")
                        except:
                            st.write(
                                "Something went wrong, please try to reformulate your "
                                "question"
                            )
            file_manager = FileManager(UPLOAD_DIR, DOWNLOAD_DIR)
            file_manager.run()

    else:
        st.write("version 0.0.1")
        st.markdown(
            '<p style="font-size:48px;text-align:center;">Agents execute code. Use Baio in a container</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            BaIO: Bridge to biological databases.
            Query NCBI, Ensembl, and ANISEED using natural language.
            Annotate your files with GO terms, Ensembl IDs, and RefSeq IDs.
            Explore and analyze your local data with AI assistance.
            """
        )
        st.markdown(
            '<p style="font-size:24px;text-align:center;">To use the <b>NCBI, ANISEED\
            </b> and <b>Local file explorer</b> you must provide a valid OpenAI API key</p>',
            unsafe_allow_html=True,
        )

        from baio.src.non_llm_tools import go_file_tool

        go_file_annotator_file_manager = FileManager(UPLOAD_DIR, DOWNLOAD_DIR)
        st.write("Local GO agent")
        with st.expander("Instructions"):
            st.markdown(read_txt_file(go_file_annotator_instruction_txt_path))
        uploaded_file = st.file_uploader("Upload your file below:", type=["csv"])
        if uploaded_file:
            st.write("You've uploaded a file!")
            save_uploaded_file(uploaded_file, UPLOAD_DIR)
            go_file_annotator_file_manager = FileManager(UPLOAD_DIR, DOWNLOAD_DIR)

        file_path = go_file_annotator_file_manager.select_file_preview_true(
            key="file_path"
        )

        with st.form("form_for_file_annotator_agent"):

            input_file_gene_name_column = st.text_area(
                "Enter gene name column:", "gene_name"
            )
            submitted = st.form_submit_button("Submit")

            if submitted:
                with get_openai_callback() as cb:

                    result = go_file_tool(file_path, input_file_gene_name_column)
                    go_file_annotator_file_manager = FileManager(
                        UPLOAD_DIR, DOWNLOAD_DIR
                    )

                    try:
                        go_file_annotator_file_manager.preview_file(result[1])
                        st.markdown(
                            go_file_annotator_file_manager.file_download_button(
                                result[1]
                            ),
                            unsafe_allow_html=True,
                        )
                        st.info(cb.total_cost)
                    except TypeError:
                        st.info(result)

        file_manager = FileManager("./baio/data/output/", "./baio/data/upload/")
        file_manager.run()

        st.markdown(
            """

        This is an application connecting the users questions and data to the internet
        and a coding agent.
        (Agent: an autonomous computer program or system that is designed to perceive
        its environment, interprets it, plans actions and executes them in order to
        achieve a defined goal.)
        It connects Large Language Models, such as OpenAI's ChatGPT, to public databases
        such as NCBI, Ensembl, and ANISEED, as well as the user's local files.
        BaIO allows users to query these databases with natural language and annotate
        files with relevant information, including GO terms, Ensembl IDs, and RefSeq IDs.
        BaIO is built on the Python LangChain library and various tools developed by
        myself and the user interface is rendered with Streamlit.
        """
        )
        st.markdown("# BaIO agent")
        st.markdown(read_txt_file(aniseed_instruction_txt_path))
        st.markdown("# Local GO Agent")
        st.markdown(read_txt_file(go_file_annotator_instruction_txt_path))
        st.markdown("# Local file agent")
        st.markdown(read_txt_file(csv_instruction_txt_path))
        st.markdown("# LICENSE")
        st.markdown(read_txt_file(LICENSE_path))


if __name__ == "__main__":
    app()
