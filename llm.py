import os

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


## llm 함수 정의 =====================================================================
def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    return llm

## database 함수 정의 =================================================================
def get_database():
    api_key = os.getenv('PINECONE_API_KEY')

    ## embedding ====================================================
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')

    ## vector store: 파인콘에 저장된 인덱스 정보 가져오기 ===========
    pc = Pinecone(api_key=api_key)

    ## 파인콘에 생성한 인덱스명
    index_name = 'law'

    database = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
    )

    return database


## retrievalQA 함수 정의 =======================================
def get_retrievalQA():
    database = get_database()

    ## RetrievalQA =============================================
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

    # ## prompt 모델 지정
    prompt = hub.pull('rlm/rag-prompt', api_key=LANGCHAIN_API_KEY)

    ## ChatGPT의 LLM 모델 지정
    llm = get_llm()

    def format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)

    qa_chain = (
        {
            'context': database.as_retriever() | format_docs,
            'question': RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain


## [AI Message 함수 정의] ======================================
def get_ai_message(user_message):
    qa_chain = get_retrievalQA()

    # qa_chain.invoke('전세사기피해자는 누구를 말하는지 알려주세요')
    ai_message = qa_chain.invoke(user_message)
    
    return ai_message
## ==================================================================

