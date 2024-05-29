# Import necessary libraries and modules
from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import gradio as gr
import time

# Define the custom prompt template
custom_prompt_template = """
You are an AI coding assistant and your task is to solve coding problems and return code snippets based on given user's query. Below is the user's query.
Query : {query}
You just return the helpful code and related details.
Helpful code and related details:
"""

# Function to set the custom prompt
def set_custom_prompt():
    # Create a PromptTemplate object with the custom template and input variables
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=['query']
    )
    return prompt

# Function to load the model
def load_model():
    # Create a CTransformers object with the specified model parameters
    llm = CTransformers(
        model='codellama-7b-instruct.ggmlv3.Q4_K_M.bin', #Download the bin file of model from hugging face i.e (codellama-7b-instruct.ggmlv3.Q4_K_M.bin)
        model_type='llama',
        max_new_tokens=1096,
        temperature=0.2,
        repetition_penalty=1.13
    )
    return llm

# Function to create the chain pipeline
def chain_pipeline():
    # Load the model
    llm = load_model()
    # Set the custom prompt
    qa_prompt = set_custom_prompt()
    # Create an LLMChain object with the custom prompt and loaded model
    qa_chain = LLMChain(
        prompt=qa_prompt,
        llm=llm
    )
    return qa_chain

# Create the chain pipeline
llmchain = chain_pipeline()

# Function to run the bot
def bot(query):
    # Run the chain pipeline with the user's query and return the response
    llm_response = llmchain.run({"query": query})
    return llm_response

# Create a Gradio interface
with gr.Blocks(title="💻MultiLang Code Assistant") as demo:
    # Add a title to the interface
    gr.Markdown("<center>💻MultiLang Code Assistant</center>")
    # Create a chatbot widget
    chatbot = gr.Chatbot([], elem_id="chatbot", height=500)
    # Create a textbox for user input
    msg = gr.Textbox()
    # Create a clear button to clear the textbox and chatbot
    clear = gr.ClearButton([msg, chatbot])
    
    # Add a submit button
    submit = gr.Button("Submit")

    # Function to respond to user input
    def respond(message, chat_history):
        # Get the bot's response to the user's message
        bot_message = bot(message) 
        # Add the user's message and the bot's response to the chat history
        chat_history.append((message, bot_message))
        # Wait for 2 seconds before returning the response
        time.sleep(2)
        return "", chat_history
    
    # Link the submit button to the respond function
    submit.click(respond, [msg, chatbot], [msg, chatbot]) 

# Launch the Gradio interface
demo.launch() 
