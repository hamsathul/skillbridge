import streamlit as st
import os
import tiktoken
import json
import random
from openai import OpenAI
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain.schema import StrOutputParser
from langchain.callbacks import StreamlitCallbackHandler
import time
import hmac

load_dotenv(find_dotenv(), override=True)
client = OpenAI()
# Function to check if the user is logged in
def check_login():
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        st.stop()

def sign_out():
    st.session_state.logged_in = False

def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            st.session_state.logged_in = True
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding= tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Function to load questions
def load_questions(filename='questions.json'):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data['questions']


# Function to initialize session state
def initialize_session_state():
    if 'selected_questions' not in st.session_state:
        st.session_state.selected_questions = random.sample(load_questions(), 10)
    if 'current_question_index' not in st.session_state:
        st.session_state.current_question_index = 0
    if 'selected_options' not in st.session_state:
        st.session_state.selected_options = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []


import streamlit as st

# class TokenPrintHandler(BaseCallbackHandler):
#     def __init__(self):
#         super().__init__()
#         self.tokens_buffer = []
#     def on_llm_new_token(self, token: str, **kwargs) -> None:
#         self.tokens_buffer.append(token)
#         if '\n' in token or len(self.tokens_buffer) > 100:
#             formatted_tokens = ''.join(self.tokens_buffer)
#             with st.container():
#                 st.markdown(formatted_tokens)
#             self.tokens_buffer = []


def summarize(conversation_history):
    if not conversation_history:
        return "no conversation history"
    with st.spinner('Skillbridge AI..Summarizing..Calculating..Recommending..🤖'):
        human_messages = "\n".join([message['content'] for message in conversation_history])
        system_prompt = """You are a very skilled assesment counselor.  you should strictly follow the report format as mentioned below. you can find the questions and responses from the user which is mentioned below.
							{human_messages}\
							you will strictly follow the below mentioned format for the assesment report step by step. please provide a well formatted text in markdown for each sections\
							name each sections appropriatly in bold.\
							1. in this section You will provide an assesment of the users responses for each question. all questions and responses have to be assesed. DO NOT mention the question itself. the assesment should be in bullet points in maximum 10 words.
							2. in this section you will asses if this person has an enterpreuner mindset or employee mindset from the data provided. be brief but not exceeding 50 words.
							3. in this section you will provide a  percentage score for both persons employee mindset and enterpreuner mindset in bold format.
							please remember that the scores for both mindsets have to be given and the total of the scores should always be 100. 
                            4. in this seciton you will make only one reccomendation and give a brief description about the program. if the user has a HIGHER ENTERPREUNER MINDSET SCORE you will reccomend skillbridge bootcamp for budding enterpreuners\
                        	else you will reccomend ignite platform for projects, internships and career guidance."""
        message_placeholder = st.empty()
        bullet_points = ""
        for response in client.chat.completions.create(
			model="gpt-3.5-turbo",
			messages=[
				{"role": "system","content": system_prompt}
			],
			stream=True,
			max_tokens=350
		):
            bullet_points += (response.choices[0].delta.content or "")
            message_placeholder.markdown(bullet_points + "|")
        message_placeholder.markdown(bullet_points)
        

# Function to display a question
def display_question(question, options):
    st.subheader(f"Question {st.session_state.current_question_index + 1}")

    # Display question text
    st.write(question)

    # Display options on new lines
    options_text = "<br>".join([f"({option['value']}) {option['option']}" for option in options])
    st.write(options_text, unsafe_allow_html=True)

def submit(response, current_question):
    
    if response and response.upper() in {'A', 'B', 'C', 'D'}:
        # Append user's question and response to conversation history
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": f"Question: {current_question['question']}\nOptions: {', '.join([option['option'] for option in current_question['options']])}"
        })
        st.session_state.conversation_history.append({"role": "assistant", "content": response.upper()})

        # Increment the question index
        st.session_state.current_question_index += 1

        # Append user's response to selected_options
        st.session_state.selected_options.append(response.upper())
    else:
        warning_msg = st.warning("Please Enter only A, B, C, or D as options")
        time.sleep(5)
        warning_msg.empty()
        

def display_question_and_response():
    # Get the current question
    current_question = st.session_state.selected_questions[st.session_state.current_question_index]

    # Display the current question in chat style
    display_question(current_question['question'], current_question['options'])

    # Use st.form to handle user input
    with st.form(key=f"form_{st.session_state.current_question_index}"):
        # Use st.text_input for user response
        response = st.text_input("Your response (Enter A, B, C, or D):")

        # Use st.form_submit_button to handle form submission
        submit_button = st.form_submit_button("Submit Response")
        if submit_button:
            submit( response, current_question)
            st.rerun()
            
def redirect_to_login():
    """Redirects the user to the login page."""
    # Clear the session state
    for key in st.session_state:
        del st.session_state[key]

    # Redirect to the login page
    st.session_state["page"] = "login"

def main():
    check_login()
    initialize_session_state()
    with st.container():
        st.markdown("<h1 style='text-align: center; background: linear-gradient(to bottom, #000099 0%, #6600ff 87%);'>SKILLBRIDGE AI ASSESMENT 🤖</h1>", unsafe_allow_html=True)
        st.subheader("", divider='rainbow')
    instructions ="""
Instructions:    
    - Questions have 4 options.
    - Type the option you choose.
    - Press submit for the next quesion.
    - Answer all questions.
    - AI will assess the responses.
    - AI will provide a report.
    - Signout by pressing the signout button.    
For any queries contact the skillbridge team.
"""
    with st.sidebar:
        with st.container():
            st.code(instructions)
        st.button("Sign Out", on_click=redirect_to_login)
    if st.session_state.current_question_index < len(st.session_state.selected_questions):
        display_question_and_response()
    else:
        st.empty()
        if st.session_state.selected_options:
            summarize(st.session_state.conversation_history)            
        else:
            st.warning("No responses recorded. Please answer the questions to generate a summary.")


if __name__ == "__main__":
    
    if not check_password():
     st.stop()
    
	        #----------------------Hide Streamlit footer----------------------------
    hide_st_style = """
	<style>	
    	MainMenu {visibility: hidden;}
		footer {visibility: hidden;}
		#header {visibility: hidden;}
	</style>
		"""
    st.markdown(hide_st_style, unsafe_allow_html=True)
    main()

