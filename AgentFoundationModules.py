#Before run python files, remember to select a python interpret using: ctrl+shift,+p
#To solve the imported module isn't resolved problem, if not, then check if its installed
import os
#os.path.exists(Ex:"file.txt"), check if file exists or not
#os.path.isfile("file.txt"), check if its a file
#os.listdir("."), lists all files/folders in current directory
#[f for f in os.listdir("." if os.path.isfile(f))], list only files
#os.mkdir("path/to/nested/folder",exist_okay=True), create nested directories
#os.remove("file.txt"),remove file. os.rmdir("folder"),remove directory
#path=os.path.join("folderName","subfolderName","file.txt"), join path:folderName\subfolderName\file.txt


import dotenv 
#Add .env file to git.ignore so API keys will be be leaked.
#reads key-value pairs(name= value in .env file) from a .env file and set them as environment variables.
#load_dotenv(), read variables from a .env file and set them in os.environ,
#when you try to get acceess to them,  use os.getenv("name").
#To overrride environment variables that has the same name, use override=True as parameter.

import json
#Javascript Object Notation
x={
    "name":"My name",#Key have to be string, and must be in camelCase.
    "age":"12"
}

import requests
#response=requests.get(url), retrieve information from server
#response.status_code, check http status, when handling errors
#200:success, 404: page not found, 500: server error, 429: Too many requests
#response.text, get raw text response, used when reading html pages or text APIs.
#response.content, as bytes(for image/files)
#response2=requests.post(url,json={"key":"value"})
#response3=requests.get(url,params={"q":"python","sort":"stars"})
#response3 result in url: origin.concat(?q=python&sort=stars)
#response4=requests.get(url,headers={"User-Agent":identify your client, "Authentization":authentization token,"Content-Type":Format of data being sent})


import pypdf
#Extract text:
"""
with open("Name.pdf","rb") as file: #rb:read binary?
    reader=pypdf.Pdfreader(file)
    num_pages=len(reader.pages)
    all_text=""
    for page in reader.pages: #extract text from all pages
        all_text+=page.extract_text()
    
"""
#Many other uses but i haven't used any so.

import gradio as gr
#simple input/output
"""
import gradio as gr

def greet(name):
    return f"Hello {name}!"

# Create interface
demo = gr.Interface(
    fn=greet,              # Function to call
    inputs="text",         # Input type
    outputs="text",        # Output type
    title="Greeting App",  # Title
    description="Enter your name and get a greeting"
)

demo.launch()  # Launch web app
""" #Create a web app at http://localhost: 7860? By description, you type your name and get greeting.
#Sentiment Analysis:
"""
from transformers import pipeline

# Load pre-trained model
classifier = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = classifier(text)[0]
    label = result['label']
    score = result['score']
    return f"Sentiment: {label}(next line operator)Confidence: {score:.2%}"

demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(label="Enter text to analyze", lines=3),
    outputs=gr.Textbox(label="Analysis Result"),
    examples=[
        ["I love this product! It's amazing!"],
        ["This is the worst experience ever."],
        ["It's okay, nothing special."]
    ],
    title="Sentiment Analysis"
)

demo.launch()
"""



#To use llama(locally and free), we should install ollama first.
# And then write "ollama pull llama3.2" in terminal.
#OLLAMA_BASE_URL = "http://localhost:11434/v1"
#For ollama, because it is running locally, the api_key just put "anything"(OpenAI)

from openai import OpenAI
#provide an access to OpenAI REST API.
#Use Response API
client=OpenAI(api_key=os.getenv(""),base_url=" ") #--Create a object. (Model could also use deepseek, llama)
response=client.responses.create(
    model="gpt-4o-mini",#or some other models
    #Required
    instruction="Your instruction for the model that is used as system prompt",
    #instruction are also used as a part of the context window
    input="This is user's input",
    #required
    text={"format":{"type":"text"}},# or "type":"json_object"
    output={"type":"text",
            "schema":{
                "type":"object",
                "properties":{
                    "key":{"ype":"string"
                    }
                }
            }
        },
    truncation="auto",#or "disabled", this mean cutting off data,like 3.9->3(Also for string) 
    #With videos
    img_url="Image url"
    input=[
        {"role":"user",
         "content":[
            {"type":"input_image","image_url":f"{img_url}"}
            ]
        }
    ]
)
#Or use Chat Completions API
completion=client.chat.completions.create(
    model="gpt-4o-mini",#required
    message=[
        {"role":"system","content":"set the bahavior, constraints/rules, format instruction"},
        #Ex: You are a sarcastic and witty assistant. Always respond in JSON format with keys: 'answer', 'confidence', 'sources'
        {"role":"user","content":"question/requests"},
        #Ex:Considering quantum physics, explain entanglement.
        """Ex: I need help with my Python project. 
        I'm building a web scraper but getting errors. 
        Here's my code: [code here]
        The error is: [error message]"""
        {"role":"assistant","content":"Assistance response(for consersation history)"}
        #Mars is the 4th planet from the Sun(when user ask tell me about mars)
    ]
)
#!Just an interesting fact, temperature controls teh randomness or creativity of the output
#or multi-turn:
"""
messages = [
    {"role": "system", "content": "You are a senior Python developer helping debug code."},
    {"role": "user", "content": "My function to calculate factorial isn't working: def fact(n): return n * fact(n-1)"},
    {"role": "assistant", "content": "You're missing a base case! Add: if n <= 1: return 1"},
    {"role": "user", "content": "I added it but now getting recursion error for fact(1000)"},
    {"role": "assistant", "content": "Python has recursion limits. Use iterative approach or sys.setrecursionlimit()"},
    {"role": "user", "content": "Can you rewrite it iteratively?"}
]"""

#globals().get("name")-> a way to access global variables
#Ex: my_var="hello", value= globals().get("my_var")->return "hello"
#Ex: 

