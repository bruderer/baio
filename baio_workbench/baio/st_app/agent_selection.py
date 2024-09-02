import streamlit as st


class MyAgents:
    def __init__(self):
        self.aniseed_agent = "Aniseed agent"
        self.baio_agent = "BaIO agent"
        self.go_file_annotator = "Local GO agent"
        self.file_chatter = "Local file agent"

    def initialise_agent_selection(self):
        return st.radio(
            "Choose an agent:",
            [
                self.aniseed_agent,
                self.baio_agent,
                self.go_file_annotator,
                self.file_chatter,
            ],
        )
