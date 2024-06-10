import streamlit as st
import time
st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# 欢迎使用地震智问系统demo! 👋")

st.sidebar.success("选择功能")

st.markdown(
    """
    基于大规模语言模型和人工智能技术,地震智问系统提供了全方位的地震数据处理和分析功能，它由三个主要模块组成:
    ### 地震数据分析可视化展示
    - 地震数据来源：()
    - 地震历史数据分析可视化展示：对原始地震数据进行清洗、处理和深入分析。它包括数据预处理、特征工程等环节,以确保数据质量。用户可以直观地观察和探索地震数据的各种模式和规律。
    ### 基于三种不同神经网络的地震预测:
    - 利用 Spark 分布式计算框架的强大并行处理能力,该模块在大规模地震数据集上训练和评估了三种先进的神经网络模型，BP神经网络、LSTM循环神经网络和Transformer。
    ### 智能问答语音交互助手
    - 结合本地知识库的智能ai问答助手
    - 语音助手，更方便的智能交互
"""
)

with st.expander("联系我们"):
    with st.form(key='contact', clear_on_submit=True):
        email = st.text_input('邮箱')
        st.text_area("问题", "在此处填入遇到的困难，有问题联系我们的小弟--郑宝炜")

        submit_button = st.form_submit_button(label='发送信息')

def cook_breakfast():
    msg = st.toast('正在建立连接...')
    time.sleep(1)
    msg.toast('正在发送...')
    time.sleep(1)
    msg.toast('已收到您的来信!', icon = "❤")

if submit_button:
    cook_breakfast()