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
