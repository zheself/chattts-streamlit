import ChatTTS
from IPython.display import Audio

chat = ChatTTS.Chat()
chat.load_models(compile=False) # Set to True for better performance

texts = ["郑宝伟是我的儿子",]

wavs = chat.infer(texts, )

torchaudio.save("output1.wav", torch.from_numpy(wavs[0]), 24000)