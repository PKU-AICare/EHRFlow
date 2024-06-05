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
    st.set_page_config(page_title="EHRFlow:您的专属医疗结构化数据分析智能助理", page_icon="🤖", layout="wide")
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
            <h4>🤖您的专属医疗结构化数据分析智能助理🥰</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("<h1 style='text-align:center;font-family:Georgia'>⚙️ EHRFlow </h1>", unsafe_allow_html=True)
        st.markdown("""一个稳定、可控、可扩展的自助式EHR数据分析平台。该系统旨在深入解决医生在实际工作中针对电子健康记录（EHR）的数据问答和数据分析需求，
                    融合了大语言模型Agent框架和本地工具类，以增强医生与数据分析工具的交互效率和准确性，从而推动医疗决策的智能化和个性化医疗服务的发展。\n""")

        st.markdown("-------")
        st.markdown("<h1 style='text-align:center;font-family:Georgia'>🌟Features</h1>", unsafe_allow_html=True)
        st.markdown(
            " - 🧾 自助式分析 - 大语言模型深度赋能，智能驱动自助式分析。")
        st.markdown(" - 🤑 数据问答- EHR 数据内容高度掌控，玩转 EHR 数据")
        st.markdown(
            " - 🧾 智能决策 - 深入分析和挖掘医疗数据，高效完成医疗智能决策")
        st.markdown("-------")
        st.markdown("<h1 style='text-align:center;font-family:Georgia'>🧾 How to use?</h1>",
                    unsafe_allow_html=True)
        st.markdown(
            "1. Enter your OpenAI API key below🔑")
        st.markdown("2. Upload Your EMR(csv) files📄")
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
    _init_mes = ("您好，我是小雅，一个擅长处理医疗结构化数据的智能助理。🥳在提交完EHR数据之后，您可以向我尽情提出关于EHR"
                 "医疗数据的问题或者处理要求，我会根据您的需求帮您处理数据，返回相应结果辅助您更好地进行医疗数据分析。现在有什么可以帮您的吗？")

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
    model = ChatOpenAI(model='gpt-4-turbo-preview', temperature=0, model_kwargs={"seed": 42})
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
