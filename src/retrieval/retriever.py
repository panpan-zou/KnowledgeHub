from src.vectorstore.chroma import ChromaService


chroma = ChromaService()

def get_retriever():
    return chroma.chroma.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )






















