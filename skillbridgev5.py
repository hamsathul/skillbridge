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



class TokenPrintHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.tokens_buffer = []

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # Accumulate tokens in the buffer
        self.tokens_buffer.append(token)

        # Check if the buffer contains a new line or reaches a certain size
        if '\n' in token or len(self.tokens_buffer) > 70:
            # Format and print the buffered tokens
            formatted_tokens = ''.join(self.tokens_buffer)
            st.markdown(formatted_tokens)
            # Reset the buffer
            self.tokens_buffer = []

def summarize(conversation_history):
    llm = ChatOpenAI(max_tokens=300, streaming=True)
    if not conversation_history:
        return "no conversation history"
    # st.subheader("Assessment Summary")
    human_messages = "\n".join([message['content'] for message in conversation_history])
    system_prompt = PromptTemplate(input_variables=["human_messages"],template = ("You are a very skilled assesment counselor. you will asses the data provided in \
    {human_messages}\
    you will strictly follow the below mentioned format for the assesment report step by step. please provide a well formatted text in markdown for each sections\
    name each sections appropriatly in bold.\
   1. in this section You will summarize the responses that the user has given for the questions in five bullet points not exceeding 10 words\
    2. in this section you will asses if this person has an enterpreuner mindset or employee mindset from the data provided. be brief and concise not exceeding 50 words \
    3. in this section you will provide a  percentage score for the persons employee mindset and enterpreuner mindset in bold format\ """))
    bullet_chain = LLMChain(llm=llm, prompt=system_prompt, verbose=False)
    bullet_points = bullet_chain.run(human_messages, callbacks=[StreamlitCallbackHandler(st.container())])
	
    score_prompt = PromptTemplate(input_variables=["bullet_points"], template=("""you are an expert in mindset assesment and career counselling.\
    the reccomendations should be given as a seperate section with bolded fonts. the reccomened programs should be bolded\
	from the result of  the previous analysis provided in \
	{bullet_points} \
    if the person have a higher enterpreuner score reccomend the user to join skillbridge bootcamp for budding entrepreuners\
    if the user have a higher employee score tell them about ignite projects and internships program.\
    you should make any one reccomendation. do not repeat the analysis just make the reccomendations dont mention the reason for why the other program is not reccomended"""))
    score_chain = LLMChain(llm=llm, prompt=score_prompt, verbose=False)
    score_answer = score_chain.run(bullet_points, callbacks=[StreamlitCallbackHandler(st.container())])  

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
    st.button("Sign Out", on_click=redirect_to_login)
    with st.container():
        st.subheader("Entrepreneur vs Employee Mindset Assessment")

    if st.session_state.current_question_index < len(st.session_state.selected_questions):
        display_question_and_response()
    else:
        # Check if there are responses before displaying the summary
        if st.session_state.selected_options:
            # Display the assessment summary after the 10th question
            # total_entrepreneur_score, total_employee_score = score_response(st.session_state.selected_options)
            
             summarize(st.session_state.conversation_history)
             st.container()
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

