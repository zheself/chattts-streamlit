import streamlit as st
from http import HTTPStatus
from dashscope import Generation
import dashscope
import ChatTTS
from wave import Wave_write
import numpy as np
import re
import base64
import io
from streamlit import components

base_path = r'C:\Users\Administrator\Desktop\ChatTTS-main\ChatTTS'

def clean_text(text):
    # 去除特殊字符和换行符
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    cleaned_text = cleaned_text.replace('\n', ' ')
    return cleaned_text

def split_text(text, max_length=500):
    # 将文本分成多个段落
    paragraphs = []
    words = text.split()
    current_paragraph = []

    for word in words:
        if len(' '.join(current_paragraph + [word])) <= max_length:
            current_paragraph.append(word)
        else:
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = [word]

    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))

    return paragraphs

def call_with_stream(user_input):
    conversation_history = st.session_state.get('conversation_history', [])
    if hasattr(st, 'audio_placeholder'):
        st.audio_placeholder.empty()
    if hasattr(st, 'placeholder1'):
        st.placeholder1.empty()
    chat = ChatTTS.Chat()
    chat.load_models(source='local', local_path=base_path)
    system_prompt = "请注意以下要求:输出一整段话,不能换行;不能出现数字、字母或符号,只能使用纯中文文字,字数不能超过150个字。"
    conversation_history.append({'role': 'system', 'content': system_prompt})

    # 添加用户问题到对话历史
    conversation_history.append({'role': 'user', 'content': user_input})

    # 调用对话生成模型
    responses = Generation.call("qwen-max",
                                messages=conversation_history,
                                result_format='message',
                                stream=True,
                                incremental_output=True)
    full_content = ''  # 用于记录完整的对话内容
    current_line = ''  # 用于跟踪当前行的内容
    line_length_threshold = 80  # 设置一行的最大字符数

    placeholder2 =st.empty()
    for response in responses:
        if response.status_code == HTTPStatus.OK:
            assistant_response = response.output.choices[0]['message']
            for char in assistant_response['content']:
                current_line += char
                if len(current_line) >= line_length_threshold and char in ['。', '，', '？', '!']:
                    full_content += current_line + '\n'
                    placeholder2.chat_message("assistant").write(full_content)
                    current_line = ''
        else:
            placeholder2.chat_message("assistant").write(
                f'Request id: {response.request_id}, Status code: {response.status_code}, error code: {response.code}, error message: {response.message}')

    # 最后一次更新，如果有剩余的内容
    if current_line:
        full_content += current_line
        placeholder2.chat_message("assistant").write(full_content)
    # 显示语音生成的进度条和提示
    with st.spinner('正在生成语音播报,请稍候...'):
        progress_bar = st.progress(0)

        # 将大模型的输出转化为语音并保存
        cleaned_content = clean_text(full_content)
        paragraphs = split_text(cleaned_content)

        wavs = []
        for i, paragraph in enumerate(paragraphs):
            params_refine_text = {
                'prompt': ''
            }
            wav = chat.infer(paragraph, params_refine_text=params_refine_text)[0]
            wavs.append(wav)
            progress_bar.progress((i + 1) / len(paragraphs))

        # 合并音频数据
        combined_wav = np.concatenate(wavs)

        sample_rate = 24000
        audio_data_rescaled = (combined_wav * 28000).astype(np.int16).flatten()

        # 将音频数据转换为字节流
        wav_bytes = io.BytesIO()
        wave_write = Wave_write(wav_bytes)
        wave_write.setparams((1, 2, sample_rate, len(audio_data_rescaled), 'NONE', 'not compressed'))
        wave_write.writeframes(audio_data_rescaled.tobytes())
        wav_bytes.seek(0)

        # 将音频数据转换为 base64 编码的字符串
        audio_base64 = base64.b64encode(wav_bytes.read()).decode()

        # 在网页上展示音频播放器
        st.audio_placeholder = st.markdown(f"""
        <audio controls>
          <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
        Your browser does not support the audio element.
        </audio>
        """, unsafe_allow_html=True)

    conversation_history.append({'role': assistant_response['role'], 'content': full_content})
    st.session_state['conversation_history'] = conversation_history
    st.placeholder1 = st.success("音频文件制作完成!")

def main():
    dashscope.api_key = 'sk-afc61e5db6a0476886909c952a0acac8'  # 填入第一步获取的APIKEY
    st.title("地震信息小助手demo")

    # 初始化对话历史
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []

    # 显示之前的对话内容
    for message in st.session_state['conversation_history']:
        if message['role'] == 'user':
            st.chat_message("user").write(message['content'])
        elif message['role'] == 'assistant':
            st.chat_message("assistant").write(message['content'])

    # 获取用户输入
    user_input = st.chat_input("请输入您的问题:", key="input")

    if user_input:
        st.chat_message("user").write(user_input)
        call_with_stream(user_input)

if __name__ == '__main__':
    main()