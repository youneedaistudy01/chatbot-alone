import os

from dotenv import load_dotenv
from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import (ChatPromptTemplate, FewShotPromptTemplate,
                                    MessagesPlaceholder, PromptTemplate)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from config import answer_examples


## 환경변수 읽어오기 =====================================================
load_dotenv()

## LLM 생성 ==============================================================
def load_llm(model='gpt-4o'):
    return ChatOpenAI(model=model)

## Embedding 설정 + Vector Store Index 가져오기 ===========================
def load_vectorstore():
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

    ## 임베딩 모델 지정
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    Pinecone(api_key=PINECONE_API_KEY)
    index_name = 'law'

    ## 저장된 인덱스 가져오기
    database = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
    )

    return database


## 세션별 히스토리 저장 =================================================
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


## 히스토리 기반 리트리버 ===============================================
def build_history_aware_retriever(llm, retriever):
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    return history_aware_retriever


def build_few_shot_examples() -> str:
    example_prompt = PromptTemplate.from_template("질문: {input}\n\n답변: {answer}")

    few_shot_prompt = FewShotPromptTemplate(
        examples=answer_examples,           ## 질문/답변 예시들 (전체 type은 list, 각 질문/답변 type은 dict)
        example_prompt=example_prompt,      ## 단일 예시 포맷
        prefix='다음 질문에 답변하세요 : ', ## 예시들 위로 추가되는 텍스트(도입부)
        suffix="질문: {input}",             ## 예시들 뒤에 추가되는 텍스트(실제 사용자 질문 변수)
        input_variables=["input"],          ## suffix에서 사용할 변수
    )

    formmated_few_shot_prompt = few_shot_prompt.format(input='{input}')

    return formmated_few_shot_prompt

## 외부 JSON 사전 불러오기
import json

def load_dictionary_from_file(path='keyword_dictionary.json'):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)

## 문자열로 변환하여 dictionary_text 생성
def build_dictionary_text(dictionary: dict) -> str:
    return '\n'.join([
        f'{", ".join(v["tags"])} (): {v["definition"]} [출처: {v["source"]}]'
        for k, v in dictionary.items()
    ])


def build_qa_prompt():

    ## 4. 외부 사전 불러오기
    keyword_dictionary = load_dictionary_from_file()
    dictionary_text = build_dictionary_text(keyword_dictionary)

    ## [keyword dictionary] ##################################################
    ## 1. 기본 형태 (가장 일반적)
    ## 장점: 키 하나당 설명 하나, 단순+빠름
    ## 용도(실무 사용 예): FAQ 챗봇, 버튼식 응답
    # keyword_dictionary = {
    # "임대인": "임대인은 주택을 임차인에게 제공하고, 계약 종료 시 보증금을 반환할 의무가 있는 자입니다.",
    # "임차인": "임차인은 임대인으로부터 주택을 임차해 거주하며, 보증금을 지급하는 계약 당사자입니다.",
    # "확정일자": "확정일자는 임대차 계약서에 공적으로 날짜를 인증받는 것으로, 보증금 보호 요건입니다.",
    # }

    ## 2. 키워드 매핑 (질문형 키워드, 질문 다양성 대응)
    ## 장점: 유사한 질문을 여러 키로 분기하여 모두 같은 응답으로 연결, fallback 대응
    ## 용도(실무 사용 예): 단답 챗봇, 버튼식 FAQ 챗봇, 키워드 트리거 응답
    # keyword_dictionary = {
    # "임대인 알려줘": "🍚임대인은 주택을 임차인에게 제공하고, 계약 종료 시 보증금을 반환할 의무가 있는 자입니다.",
    # "임대인 설명해줘": "🍚임대인은 주택을 임차인에게 제공하고, 계약 종료 시 보증금을 반환할 의무가 있는 자입니다.",
    # "임차인": "🍜임차인은 임대인으로부터 주택을 임차해 거주하며, 보증금을 지급하는 계약 당사자입니다.",
    # }

    ## 3. 키워드 + 태그 기반 딕셔너리
    ## 장점: 확장성, 분류 가능
    ## 용도(실무 사용 예): 검색 인덱스 생성, 카테고리 필터링, 하이라이팅 등
    ## 사용 방식 : 키워드 -> 응답 내용 + 출처 + 분류 태그
    # keyword_dictionary = {
    #     '임대인': {
    #         'definition': '전세사기피해자법 제2조 제2항에 따른 임대인 정의입니다.',
    #         'source': '전세사기피해자법 제2조',
    #         'tag': ['법률', '용어', '기초'],
    #     },
    #     '보증금': {
    #         'definition': '보증금은 계약 종료 시 임대인이 반환해야 할 금전입니다',
    #         'source': '주택임대차보호법 제3조',
    #         'tags': ['금융', '보호', '우선변제'],
    #     }
    # }

    # dictionary_text = '\n'.join([
    #     f'{k} (", ".join(v["tags"])): {v["definition"]} [출처: {v["source"]}]' 
    #     for k, v in keyword_dictionary.items()
    # ])

    system_prompt = (
    '''[identity]
- 당신은 전세사기피해 법률 전문가입니다.
- [context]와 [keyword_dictionary]를 참고하여 사용자의 질문에 답변하세요.
- 답변에는 해당 조항을 '(XX법 제X조 제X항 제X호, XX법 제X조 제X항 제X호)' 형식으로 문단 마지막에 표시하세요.
- 항목별로 표시해서 답변해주세요.
- 전세사기피해 법률 이외의 질문에는 '답변할 수 없습니다.'로 답하세요.

[context]
{context} 

[keyword_dictionary]
{dictionary_text}
'''    
    )

    formmated_few_shot_prompt = build_few_shot_examples()

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ('assistant', formmated_few_shot_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    ).partial(dictionary_text=dictionary_text)

    return qa_prompt


## 전체 chain 구성 =================================================
def build_conversational_chain():
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

    ## LLM 모델 지정
    llm = load_llm()

    ## vector store에서 index 정보
    database = load_vectorstore()
    retriever = database.as_retriever(search_kwargs={'k': 2})

    history_aware_retriever = build_history_aware_retriever(llm, retriever)

    qa_prompt = build_qa_prompt()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key='answer',
    ).pick('answer')

    return conversational_rag_chain


## AI Message ===========================================================
def stream_ai_message(user_message, session_id='default'):
    qa_chain = build_conversational_chain()

    ai_message = qa_chain.stream(
        {'input': user_message},
        config={'configurable': {'session_id': session_id}},        
    )

    print(f'대화 이력 >> {get_session_history(session_id)} \n😎\n')
    print('=' * 50 + '\n')
    print(f'[stream_ai_message 함수 내 출력] session_id >> {session_id}')

    ##########################################################################
    ## 벡터 DB에서 검색된 문서 확인
    retriever = load_vectorstore().as_retriever(search_kwargs={'k': 2})
    search_results = retriever.invoke(user_message)

    print(f'\nPinecone 검색 결과 >> \n{search_results[0]} \n\n{search_results[1]}')
    #########################################################################

    return ai_message

