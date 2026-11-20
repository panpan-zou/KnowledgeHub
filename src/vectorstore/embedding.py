""" embedding 模型 """
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

embed_name = os.getenv("EMBEDDING_NAME")
embedding = HuggingFaceEmbeddings(
    model_name=embed_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
    query_encode_kwargs={'normalize_embeddings': True},
    show_progress=False,
)


