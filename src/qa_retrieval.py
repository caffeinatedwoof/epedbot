import os

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def qa_retrieval_from_eped_guide(query):
    # Load vectore store saved locally
    path = "../data/vector_data/"
    if os.path.exists(path):
        vector_store = FAISS.load_local(path, OpenAIEmbeddings())
    else:
        print(
            f"Missing files. Upload index.faiss and index.pkl files to {path} directory first"
        )

    system_template = """Use the following pieces of context to answer the users question.\
    No matter what the question is, you should always answer it in the context of the e-Pedagogy Guide.\
    If possible, point the user to the page of guide where you obtained the information from. \
    Even if the question does not end in a question mark, you should still answer it as if it were a question.\
    If the question is not related to the e-Pedagogy Guide, reply "This falls outside the scope of the e-Pedagogy Guide".\
    ----------------
    {summaries}"""

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    # Initialize the chain
    chain_type_kwargs = {"prompt": prompt}
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 2}
        ),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )
    result = chain(query)
    return result
