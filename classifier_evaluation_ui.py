import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
os.environ["NVIDIA_API_KEY"] = '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'
def read_questions(file_path):
    question_categories = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 假设每一行的格式为 "问题——类别"
            question = line.strip().split('——')[0]
            category = line.strip().split('——')[1]
            question_categories.append((question,category))
    return "\n\n".join(f"question_category:{question_category}" for question_category in question_categories),question_categories
def ragchain_result(chain,context,query):
    result = ''
    chunk_stream = chain.invoke({
        'question': query,
        'context':context,
    })
    for chunk in chunk_stream:
        result = result + chunk
    return result,chunk_stream
def question_classifier(question,context):
    question_classify_prompt = ChatPromptTemplate.from_messages(
        [
            (
                'system',
                """
                You are an intelligent classifier to classifying questions into one of the following four categories: 'personal background', 'research interest', 'publication' and 'recruitment'. You must learn the correspondence of the question and its category in the context and then classify.
                Context:{context}  
                Question: {question}
                Output format: Only output one phrase in the format of <class name>
                """
            ),
        ]
    )
    question_classify_llm = ChatNVIDIA(model='meta/llama-3.2-3b-instruct')
    question_classify_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | question_classify_prompt
            | question_classify_llm
            | StrOutputParser()
    )
    result, chunk_stream = ragchain_result(question_classify_chain,context,question)
    return result.strip("'").strip("<").strip(">")

generated_questions_text,generated_questions=read_questions('./generated_question.txt')
#手动评估问题
#UI
st.set_page_config(layout="wide", page_title="classifier_evaluation")
st.title("Classifier Evaluation System")
st.header("You should input a question about the researchers in PolyU and you will get the category.")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": "Hi, Guys!"}]
for msg in st.session_state.messages:
    if msg["role"] == "system":
        st.chat_message(msg["role"], avatar="polyu.jpg").write(msg["content"])
    else:
        st.chat_message(msg["role"]).write(msg["content"])
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    st.chat_message("user").write(input)
    with st.spinner("Thinking..."):
        category = question_classifier(input, generated_questions_text)
        st.session_state.messages.append({"role": "system", "content": category})
        print(st.session_state.messages)
        st.chat_message("system", avatar="polyu.jpg").write(category)
