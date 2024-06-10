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

dashscope.api_key = 'sk-afc61e5db6a0476886909c952a0acac8'  # 填入第一步获取的APIKEY
base_path = r'C:\Users\Administrator\Desktop\chattts\ChatTTS-main\ChatTTS'


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
    chat = ChatTTS.Chat()
    chat.load_models(source='local', local_path=base_path)
    system_prompt = "请注意以下要求:输出一整段话,不能换行;不能出现数字、字母或符号,只能使用纯中文文字,字数不超过120个字。"
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

    for response in responses:
        if response.status_code == HTTPStatus.OK:
            assistant_response = response.output.choices[0]['message']
            full_content += assistant_response['content'] + '\n'  # 记录对话内容

            for char in assistant_response['content']:
                current_line += char
                if len(current_line) >= line_length_threshold and char in ['。', '，', '？', '!']:
                    st.write(current_line)
                    current_line = ''

        else:
            st.write('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))

    if current_line:
        st.write(current_line)

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
        st.markdown(f"""
        <audio controls>
          <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
        Your browser does not support the audio element.
        </audio>
        """, unsafe_allow_html=True)

    conversation_history.append({'role': assistant_response['role'], 'content': full_content})
    st.session_state['conversation_history'] = conversation_history
    st.success("音频文件制作完成!")


def main():
    st.title("地震信息小助手demo")

    # 显示之前的对话内容
    conversation_history = st.session_state.get('conversation_history', [])
    for message in conversation_history:
        if message['role'] == 'user':
            st.write(f"用户: {message['content']}")
        elif message['role'] == 'assistant':
            st.write(f"助手: {message['content']}")

    # 将输入框和提交按钮放在 st.form 中
    with st.form(key='input_form'):
        user_input = st.text_input("请输入您的问题:")
        submit_button = st.form_submit_button(label='提交')

    if submit_button:
        if user_input:
            call_with_stream(user_input)
        else:
            st.warning("请输入您的问题!")


if __name__ == '__main__':
    main()