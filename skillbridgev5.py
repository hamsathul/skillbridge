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


class TokenPrintHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.tokens_buffer = []

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # Accumulate tokens in the buffer
        self.tokens_buffer.append(token)

        # Check if the buffer contains a new line or reaches a certain size
        if '\n' in token or len(self.tokens_buffer) > 40:
            # Format and print the buffered tokens
            formatted_tokens = ''.join(self.tokens_buffer)
            st.write(formatted_tokens)
            # Reset the buffer
            self.tokens_buffer = []


client = OpenAI()
llm = ChatOpenAI(max_tokens=1000, streaming=True, callbacks=[TokenPrintHandler()])
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
    # if 'logged_in' not in st.session_state:
    #     st.session_state.logged_in = True

# Function to ask a question
# def ask_question(question, options):
#     prompt = f"{question}\nOptions:\n(A) {options[0]['option']}\n(B) {options[1]['option']}\n(C) {options[2]['option']}\n(D) {options[3]['option']}\nYour response: "

#     response_content = client.chat.completions.create(
#         model="text-davinci-002",
#         prompt=prompt,
#         temperature=0.7,
#         max_tokens=150
#     )['choices'][0]['text']
#     print(response_content)
#     return {'role': 'user', 'content': response_content}


# Function to summarize conversation history
def summarize_conversation_history(conversation_history):
    if not conversation_history:
        return "No conversation history available."

    human_messages = "\n".join([message['content'] for message in conversation_history])

    template = "You are a very skilled assesment counselor. you will asses the data provided and suggest\
    skillbridge boot camp if the person has a higher enterpreuner score or you will suggest \
        ignite platform for projects, internships and career guidance.\
    . you will strictly follow the below mentioned format for the assesment report step by step\
   1. You will summarize the responses that the user has given for the questions in five bullet points\
    2. you will asses if this person has an enterpreuner mindset or employee mindset from the data provided. \
    3. you will provide a  percentage score for the persons employee mindset and enterpreuner mindset in bold format\
    4. if the person has a higher enterpreuner mindest score percentage you will recommend SKILLBRIDGE to join the boot camp on enterpreunership \
    5. if the person has a higher employee mindet score percentage you will recommend ignite platform for projects and internships \
    "
    messages = [
    	SystemMessage(
            content=template),
    	HumanMessage(
            content=human_messages ),
	]
    final_token_sent = num_tokens_from_string(human_messages) + num_tokens_from_string(template)
    final_token_sent_amt = 0.0030 * final_token_sent / 1000 * 85
    print(f'token sent: {final_token_sent} \nsent amount: INR{final_token_sent_amt}\n')
    summary = llm(messages)
    return summary.content

# Function to score the response
def score_response(selected_options):
    scoring = {'A': (100, 0), 'B': (75, 25), 'C': (25, 75), 'D': (0, 100)}
    
    # Calculate scores based on the distribution of options
    total_entrepreneur_score = sum(scoring[option][0] for option in selected_options)
    total_employee_score = sum(scoring[option][1] for option in selected_options)

    return total_entrepreneur_score, total_employee_score

# Function to display a question
def display_question(question, options):
    st.subheader(f"Question {st.session_state.current_question_index + 1}")

    # Display question text
    st.write(question)

    # Display options on new lines
    options_text = "<br>".join([f"({option['value']}) {option['option']}" for option in options])
    st.write(options_text, unsafe_allow_html=True)

# Function to display the assessment summary
def display_summary(total_entrepreneur_score, total_employee_score, selected_options, conversation_history):
    st.subheader("Assessment Summary")
    final_token_recd = num_tokens_from_string(summarize_conversation_history(st.session_state.conversation_history))
    recvd_amt = 0.0060 * final_token_recd / 1000 * 85
    print(f'token received: {final_token_recd} \nsent amount: INR {recvd_amt}\n')
    # st.write(f"Total Entrepreneur Score: {total_entrepreneur_score/10:.2f} out of 100")
    # st.write(f"Total Employee Score: {total_employee_score/10:.2f} out of 100")

    # st.subheader("Questions and Responses:")
    # for i, entry in enumerate(conversation_history, start=1):
    #     role = entry['role']
    #     content = entry['content']

    #     if role == 'user':
    #         st.write(f"{i}. User: {content}")
    #         st.write(f"   Selected Option: {selected_options[i-1]['content']}")
    #     elif role == 'assistant':
    #         st.write(f"{i}. Assistant: Option {content}")

    # st.subheader("Conversation Summary")
    # st.markdown(summary)


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
    # st.rerun()


def main():
    check_login()
    initialize_session_state()
    st.button("Sign Out", on_click=redirect_to_login)
    with st.container():
        st.subheader("Entrepreneur vs Employee Mindset Assessment")
    # if st.button("Sign Out", on_click=redirect_to_login):
    #     redirect_to_login()


    # if st.button("Sign Out"):
    #  sign_out()
    #  redirect_to_login()


    # st.title("SKILLBRIDGE AI ASSESSMENTS")

    # Check if all questions have been answered
	
    if st.session_state.current_question_index < len(st.session_state.selected_questions):
        display_question_and_response()
    else:
        # Check if there are responses before displaying the summary
        if st.session_state.selected_options:
            # Display the assessment summary after the 10th question
            total_entrepreneur_score, total_employee_score = score_response(st.session_state.selected_options)
            display_summary(total_entrepreneur_score, total_employee_score, st.session_state.selected_options, st.session_state.conversation_history)
        else:
            st.warning("No responses recorded. Please answer the questions to generate a summary.")


if __name__ == "__main__":


    if not check_password():
     st.stop()
    main()
    
	        #----------------------Hide Streamlit footer----------------------------
    hide_st_style = """
	<style>	
    	MainMenu {visibility: hidden;}
		footer {visibility: hidden;}
		#header {visibility: hidden;}
	</style>
		"""
    st.markdown(hide_st_style, unsafe_allow_html=True)
