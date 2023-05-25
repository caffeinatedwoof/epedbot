import streamlit as st
from qa_retrieval import qa_retrieval_from_eped_guide


def app():
    # Create text input for the query
    question = st.text_input("Your Query")

    # If the query is not empty, process it
    if st.button("Submit"):
        if question:
            st.write("Processing your query...")
            result = qa_retrieval_from_eped_guide(question)
            st.write("Answer:", result["answer"])
            for i, entry in enumerate(result["source_documents"]):
                st.write(f"Source {i+1}")
                page_num = result["source_documents"][i].metadata["page"]
                st.write(f"This is taken from page {page_num} of the e-Pedagogy guide")
                st.write(result["source_documents"][i].page_content)
        else:
            st.write("Please enter a query.")


# Run the app
if __name__ == "__main__":
    app()
