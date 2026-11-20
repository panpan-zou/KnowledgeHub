''' 文本加载模块 '''

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader, TextLoader
from langchain_core.documents import Document
from pathlib import Path


def read_path(dirs):
    """校验目录是否存在，校验是否为目录。并返回目录下的文件列表"""
    path = Path(dirs)
    if not path.exists():
        raise FileNotFoundError(f'{dirs} 目录不存在')
    if not path.is_dir():
        raise FileNotFoundError(f'{path} 不是一个目录')

    return [file for file in path.iterdir()]


def create_loader(file):
    """按不同类型的文件，用相应的方法读取 document"""
    suffix = Path(file).suffix
    # 普通文本和 Markdown 在本课都按 UTF-8 纯文本读取。
    if suffix in {".txt", ".md"}:
        return TextLoader(str(file), encoding="utf-8")
    # mode="page" 让 PDF 的每一页生成一个 Document。
    if suffix == ".pdf":
        return PyPDFLoader(str(file), mode="page")
    if suffix == ".docx":
        return Docx2txtLoader(str(file))
    # utf-8-sig 可以兼容带 BOM 的 UTF-8 CSV 文件。
    if suffix == ".csv":
        return CSVLoader(file_path=str(file), encoding="utf-8-sig")
    raise ValueError(f'{file} 不支持读取该类型文件')

def require_file(file_path):
    """确认路径指向一个存在的普通文件。"""
    # 先统一转换成 Path，后面才能稳定读取后缀和检查文件。
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在：{path}")
    return path

def normalize_documents(file_path, documents):
    """统一正文、来源路径以及 page/row 编号。"""
    path = require_file(file_path)
    suffix = path.suffix.lower()
    normalized = []

    for document in documents:
        # 去掉正文首尾空白；空正文不进入后续分块流程。
        text = document.page_content.strip()
        if not text:
            continue

        # 先保留 Loader 自带字段，再覆盖为本项目统一的来源字段。
        metadata = {
            **document.metadata,
            "source": path.as_posix(),
            "filename": path.name,
            "file_type": suffix.lstrip("."),
        }

        # PyPDFLoader 的 page 是从 0 开始的页索引。
        if suffix == ".pdf" and isinstance(metadata.get("page"), int):
            metadata["page"] += 1

        # CSVLoader 的 row 是从 0 开始的数据行索引；原文件第 1 行是表头。
        if suffix == ".csv" and isinstance(metadata.get("row"), int):
            metadata["row"] += 2

        # 用规范化后的正文和 metadata 创建新的 Document。
        normalized.append(Document(
            page_content=text,
            metadata=metadata,
        ))

    if not normalized:
        raise ValueError(f"文件没有可用正文：{path}")
    return normalized


def load_documents(dirs):
    """选择现成 Loader，加载后统一 Document 数据合同。"""
    # 读取目录，返回文件列表
    file_paths = read_path(dirs)
    documents = []
    for file_path in file_paths:
        # 每个文件先选择 Loader，再调用统一的 load() 接口。
        loader = create_loader(file_path)
        loaded = loader.load()
        # 不同 Loader 的输出先规范化，再合并到同一个列表。
        documents.extend(normalize_documents(file_path, loaded))
    return documents









