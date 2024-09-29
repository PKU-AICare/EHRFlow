import streamlit as st
from langchain_core.callbacks import CallbackManager

upload_dir = "datasets"
import openai
import os

from Tools.PythonTool import xiaoyaAnalyser
from langchain_community.callbacks import StreamlitCallbackHandler
from Utils.streamlitoutcallbackhandler import ChainStreamHandler
from GraceAgent import (
    GraceExecutor,
    GraceGPT,
    GracePlanner
)
from Tools import *
from langchain_openai import ChatOpenAI, OpenAI
from dotenv import load_dotenv, find_dotenv
from Tools.Interpreter import Interpreter

import warnings

warnings.filterwarnings("ignore")


def main():
    st.set_page_config(page_title="EHRFlow: EHR Analysis Assistant", page_icon="🤖", layout="wide")
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>EHRFlow</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style='text-align: center;'>
            <h4>🤖 Your intelligent assistant for EHR data analysis 🥰</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("<h1 style='text-align:center;font-family:Georgia'>⚙️ EHRFlow </h1>", unsafe_allow_html=True)
        st.markdown("""An expandable, user-friendly, self-service EHR data analysis platform.\n""")

        st.markdown("-------")
        st.markdown("<h1 style='text-align:center;font-family:Georgia'>🌟Features</h1>", unsafe_allow_html=True)
        st.markdown(" - 🤑 Self-service Analysis - Complete control over EHR data content, mastering EHR data.")
        st.markdown(
            " - 🧾 Intelligent Decision-making - In-depth analysis and mining of medical data, efficiently achieving intelligent medical decision-making.")
        st.markdown("-------")
        st.markdown("<h1 style='text-align:center;font-family:Georgia'>🧾 How to use?</h1>",
                    unsafe_allow_html=True)
        st.markdown(
            "1. Enter your OpenAI API key below🔑")
        st.markdown("2. Upload Your EHR (CSV) files📄")
        st.markdown(
            "3. Ask a question about your data💬")
        if os.path.exists(".env"):
            _ = load_dotenv(find_dotenv())
            openai.api_key = os.environ['OPENAI_API_KEY']
            openai.base_url = os.environ['OPENAI_API_BASE']
            st.success("API key loaded from .env", icon="🚀")
        else:
            user_api_key = st.sidebar.text_input(
                label="#### Enter OpenAI API key 👇", placeholder="Paste your openAI API key, sk-", type="password",
                key="openai_api_key"
            )
            openai.api_key = user_api_key
            if user_api_key:
                st.sidebar.success("API key loaded", icon="🚀")
        st.markdown('-------')
        # st.markdown('Peking University')

    uploaded_files = st.file_uploader("Upload your EHR data here 👇:", type="csv", accept_multiple_files=True)
    if not uploaded_files:
        st.stop()

    with st.spinner("Uploading documents... This may take a while⏳"):
        for uploaded_file in uploaded_files:
            # 获取上传文件的名称
            filename = os.path.join(upload_dir, uploaded_file.name)

            # 将文件写入到本地指定路径
            with open(filename, "wb") as f:
                f.write(uploaded_file.getvalue())

    print([uploaded_file.name for uploaded_file in uploaded_files])
    _init_mes = ("Hello, I am EHRFlow, an intelligent assistant specializing in handling structured medical data. 🥳 After submitting your EHR data, you can feel free to ask me any questions or processing requests related to EHR medical data. I will process the data according to your needs and return corresponding results to assist you in better analyzing medical data. How can I assist you today?")

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": _init_mes}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": _init_mes}]

    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    st_callback = StreamlitCallbackHandler(st.container())
    interpreter = Interpreter()
    model = ChatOpenAI(model='gpt-4o-2024-08-06',
                       openai_api_base="",
                       openai_api_key="",
                       temperature=0)
    tools = [
        directory_inspection_tool,
        CSV_inspection_tool,
        finish_placeholder,
        xiaoyaAnalyser(
            model=model,
            prompt_path="./prompts/Tools",
            info_path="./Tools/xiaoya_info",
            interpreter=interpreter
        ).as_tool()
    ]
    dataset_folder_path = "./datasets"
    csv_files = [os.path.join(dataset_folder_path, uploaded_file.name) for uploaded_file in uploaded_files]
    planner = GracePlanner(model, tools, csv_files, stop=['<END_OF_PLAN>'])
    executor = GraceExecutor(
        llm=model,
        prompts_path="prompts/executor",
        tools=tools,
        work_dir='datasets',
        main_prompt_file='executor.json',
        files=csv_files,
        final_prompt_file='final_step.json',
        max_thought_steps=10,
    )
    agent = GraceGPT(planner=planner, executor=executor)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            response = agent.run(prompt, callbacks=[st_callback])
            # st.write_stream(chainStreamHandler.generate_tokens())
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
