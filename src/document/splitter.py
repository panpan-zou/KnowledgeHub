''' 文本切分 '''
import hashlib
from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHUNK_SIZE = 300
OVER_LAP = 50

# 创建一个切分 document 的工具
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVER_LAP,
    separators=['\n\n', '\n', '。', '!', '？', ';', ' ', ''],
    add_start_index=True
)

def make_id(index, source, text):
    return hashlib.sha256(f'{index}{source}{text}'.encode('utf-8')).hexdigest()[:24]

def split_document(docs):
    # 将 document 按自然段落再进行切分
    documents = splitter.split_documents(docs)
    chunks = []
    # 给每一个来源都建立一个独立计算的 chunk_id
    chunk_list = defaultdict(int)
    for document in documents:
        source = document.metadata['source']
        index = chunk_list[source]
        text = document.page_content
        # 给元数据单独添加 chunk_id
        chunks.append(Document(
            id=make_id(index, source, text),
            page_content=text,
            metadata={
                **document.metadata,
                'chunk_index': index,
            }
        ))
        chunk_list[source] += 1

    return chunks























