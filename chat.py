import streamlit as st
from dotenv import load_dotenv
from llm import get_ai_message


## 환경변수 읽어오기 ============================================
load_dotenv()

st.set_page_config(page_title='전세사기피해 상담 챗봇', page_icon='🤔')
st.title('🤔전세사기피해 상담 챗봇 ')

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

# print(f'before: {st.session_state.message_list}')

## 채팅 내용 모두 표시 ############
for message in st.session_state.message_list:
    # print('message >>', message)
    with st.chat_message(message['role']):
        st.write(message['content'])


## prompt 창(채팅 창) ############
placeholder = '전세사기피해와 관련된 궁금한 내용을 질문하세요.'
if user_question := st.chat_input(placeholder=placeholder):
    with st.chat_message('user'):
        ## 사용자 메시지
        st.write(user_question)
    
    st.session_state.message_list.append({'role': 'user', 'content': user_question})

    with st.spinner('답변 생성하는 중입니다.'):
        # ai_message = get_ai_message(user_question)

        session_id = 'user-session'
        ai_message = get_ai_message(user_question, session_id=session_id)

        with st.chat_message('ai'):
            ai_message = st.write_stream(ai_message)
        st.session_state.message_list.append({'role': 'ai', 'content': ai_message})

# print(f'after: {st.session_state.message_list}')


