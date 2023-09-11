import json
import openai
import re
import language_tool_python
import os
from torchtext.data.utils import get_tokenizer

#defining constants based on app.config
with open(os.getcwd() + "\\app.config") as settings_file:
   appSettings = json.load(settings_file)
TRAINING_FILE_PATH = os.path.abspath(appSettings["training_file_name"])
TRAINING_PROMPT = appSettings["training_prompt"]
LOWER_RANGE = int(appSettings["starting_point"])
tokenizer = get_tokenizer("basic_english")
tool = language_tool_python.LanguageTool('en-UK')
FILE_NAME = os.getcwd() + "\\" + appSettings["tone"] + "PromptCompletions.jsonl"
TONE_OF_VOICE = "professional"
API_KEY = appSettings["api_key"]
MODEL = "gpt-3.5-turbo"
openai.api_key = API_KEY
os.environ['OPENAI_API_KEY'] = API_KEY

#opens file containing prompt articles
f = open(TRAINING_FILE_PATH, encoding="utf8") #changes absolute path to relative path
lines = f.readlines()
f.close()

#creates file where training data will be written
g = open(FILE_NAME, "a") 
trainingJson = ""


for i in range(LOWER_RANGE,len(lines)):
    trainingJson = ""
    #this if statement allows delimiters to be put into the text file
    if len(lines[i]) > 10:

        try:
            #"system" messages provide the instructions for GPT, while "user" messages provide the article to be rewritten.
            tokenizedLine = tokenizer(lines[i])
            cleanedLine = ' '.join(tokenizedLine)
            trainingJson =''
            trainingJson += '{"messages": [{"role": "system", "content": "' + TRAINING_PROMPT + '"},'
            trainingJson += '{"role": "user", "content": ' + json.dumps(cleanedLine) + '},'
            
            completion = openai.ChatCompletion.create(
            model=MODEL,
            temperature = 0.81,
            messages=[
                {"role": "user", "content": TRAINING_PROMPT + cleanedLine}
            ]
            )
            cleanedCompletion = completion.choices[0].message.content
            #Each rewritten article is written to the "assistant" portion of the line.
            trainingJson += '{"role": "assistant", "content": ' + json.dumps(cleanedCompletion) + '}]}'
            trainingJson +='\n' #each example has to be on its own line
            #print(i) [uncomment this if you want to know exactly how much progress has been made]
            if i%50==0: #provides progress reports every 50 articles
                print(str(i) + " lines parsed")
            g.write(trainingJson) #writes trainingJson to the training file. this ensures that in the case of an exception, no data is lost.
        
        #rate limit errors are the most common type of error. you may also run out of funds.
        except:
            print("There may have been some issue with OpenAI's servers. Change the starting_point value to " + str(i) + " and rerun the script. Check the usage tab in your OpenAI account to ensure that you have sufficient funds.")
            break
        
trainingJson = ""    
g.close()
print("Training file successfully created")
