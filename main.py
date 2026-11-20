""" 入口交互 """

from src.document.loader import load_documents
from src.document.splitter import split_document
from src.vectorstore.chroma import ChromaService
from src.retrieval.retriever import get_retriever

from dotenv import load_dotenv
import json
import os

def menu():
    print(
'''
========= 智能知识库系统 =========
 1、添加数据到向量库
 2、查询单个问题
 0、退出
===============================
''')

def load():
    load_dotenv()
    return {
        'api_key': os.getenv('API_KEY'),
        'base_url': os.getenv('BASE_URL'),
        'model_name': os.getenv('MODEL_NAME'),
        'dir_path': os.getenv('DIR_PATH'),
    }


def main():
    data = load()
    chroma = ChromaService()

    while True:
        menu()
        choice = input('请输入->:').strip()
        if choice == '1':
            try:
                res = load_documents(data['dir_path'])
                print('读取到的 document 总数:', len(res))
                chunks = split_document(res)
                print('切分后的 chunk 数量', len(chunks))
                chroma.add_document(chunks)
                print('已成功导入!')
            except Exception as e:
                print(e)

        elif choice == '2':
            try:
                question = input('请输入问题->:').strip()
                evidence = chroma.query_question(question)
                for e in evidence:
                    print(e)
            except Exception as e:
                print(e)

        elif choice == '3':
            retriever = get_retriever()
            evidence = retriever.invoke('住宿费怎么报销')
            print(evidence)

        elif choice == '0':
            break
        else:
            print('❌️输入内容错误，请重试！')


if __name__ == '__main__':
    main()

















