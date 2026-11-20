""" chroma 维护 """

from langchain_chroma import Chroma
from dotenv import load_dotenv

from .embedding import embedding
import os
load_dotenv()

chroma_dir = os.getenv("CHROMA_PATH")
collection_name = os.environ['COLLECTION_NAME']

class ChromaService:
    def __init__(self):
        self.chroma = Chroma(
            collection_name=collection_name,
            persist_directory=chroma_dir,
            embedding_function=embedding,
            collection_configuration={'hnsw': {'space': 'cosine'}}
        )

    def add_document(self, documents:list):
        ids = [doc.id for doc in documents]
        self.chroma.add_documents(
            documents=documents,
            ids=ids,
        )

    def remove_document(self, chunk_id:list):
        self.chroma.delete(
            ids=chunk_id,
        )

    def update_document(self, chunk_id:list, document:list):

        res = self.get_document([chunk_id])
        if not res:
            raise ValueError(f'{chunk_id} 不存在!')

        self.chroma.update_documents(
            ids=chunk_id,
            documents=document,
        )

    def query_question(self, question):
        return self.chroma.similarity_search_with_score(question, k=3)

    def get_document(self, chunk_id:list):
        return self.chroma.get_by_ids(chunk_id)