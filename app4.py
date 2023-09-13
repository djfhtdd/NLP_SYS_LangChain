import os
import streamlit as st
from apikey import apikey
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text=initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # "/" is a marker to show difference 
        # you don't need it 
        self.text += token
        self.container.markdown(self.text)
    def on_llm_end(self,response, **kwargs) -> None:
        self.container.empty()
        pass

os.environ['OPENAI_API_KEY'] = apikey

st.header('🎼🎵🎶🔗 Text-to-Song Summarization')
prompt1 = st.text_input('Fill your text')
prompt2 = st.text_input('Choose your theme(s)')
submit_button=st.button("Submit")

text_template = PromptTemplate(
    input_variables = ['text'],
    template = "Summarize ```{text}```"
)

song_template = PromptTemplate(
    input_variables = ['theme','chat_history'],
    template = "```{chat_history}``` Compose the song and name the song from the summarized text using given themes as references.\
    The themes are delimited with brackets. THEME:({theme}). Show song name before lyrics.")

memory = ConversationBufferMemory(input_key='text', memory_key='chat_history')

chat_box=st.empty() 
stream_handler = StreamHandler(chat_box)

chat = ChatOpenAI(temperature = 0.8, streaming=True,callbacks=[stream_handler])
text_chain = LLMChain(llm=chat, prompt=text_template, output_key = 'sum', memory = memory)
song_chain = LLMChain(llm=chat, prompt=song_template, output_key = 'song', memory = memory)
seq_chain = SequentialChain(chains=[text_chain, song_chain], input_variables = ['text','theme'], return_all = True, verbose=True)

if prompt1 and prompt2:
    response = seq_chain({'text':prompt1, 'theme': prompt2})
    st.markdown('**📄Summary**')
    st.write(response['sum'])
    st.markdown('**🎤Lyrics**')
    st.write(response['song'])
