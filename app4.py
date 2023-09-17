import os
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 
from langchain.schema import HumanMessage
import streamlit as st

with st.sidebar:
    st.markdown(':rainbow[Model adjustment]') 
    os.environ['OPENAI_API_KEY'] = st.text_input('Enter your OpenAI API key here', type='password') or st.secrets['OPENAI_API_KEY']
    model_name = st.selectbox('Choose model', ('gpt-3.5-turbo', 'gpt-3.5-turbo-16k','gpt-4'))
    temperature = st.number_input('Choose temperature',  min_value=0.0, max_value=1.0, value= 0.8 , step= 0.05)

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

st.header('🎼🎵🎶🔗 Text-to-Song Summarization')
prompt1 = st.text_input('Fill your text')
prompt2 = st.text_input('Choose your theme(s)')
submit_button=st.button("Submit")

text_template = PromptTemplate(
    input_variables = ['text'],
    template = "Summarize ```{text}```"
)

if len(prompt2.split()) <= 15:
    song_template = PromptTemplate(
    input_variables = ['theme','chat_history'],
    template = "```{chat_history}``` Compose the song and name the song from the summarized text using given themes as references.\
    THEME:({theme}). Show song name above. Show what themes are used in each music structure at the beginning of music structure in parentheses.")
else:
    song_template = PromptTemplate(
    input_variables = ['theme','chat_history'],
    template = "```{chat_history}``` Turn into a song and name a song based on this lyrics. LYRICS: ```{theme}```")

memory = ConversationBufferMemory(input_key='text', memory_key='chat_history')

chat_box=st.empty() # here is the key, setup a empty container first
stream_handler = StreamHandler(chat_box)

chat = ChatOpenAI(model_name = model_name,temperature = temperature, streaming=True,callbacks=[stream_handler])
text_chain = LLMChain(llm=chat, prompt=text_template, output_key = 'sum', memory = memory)
song_chain = LLMChain(llm=chat, prompt=song_template, output_key = 'song', memory = memory)
seq_chain = SequentialChain(chains=[text_chain, song_chain], input_variables = ['text','theme'], return_all = True, verbose=True)

if prompt1 and prompt2:
    response = seq_chain({'text':prompt1, 'theme': prompt2})
    st.markdown('**📄Summary**')
    st.write(response['sum'])
    st.markdown('**🎤Lyrics**')
    st.write(response['song'])
