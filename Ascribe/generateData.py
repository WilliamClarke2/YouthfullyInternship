import json
import openai
import re
import language_tool_python
import os
from torchtext.data.utils import get_tokenizer

with open(os.getcwd() + "\\app.config") as settings_file:
   appSettings = json.load(settings_file)
TRAINING_FILE_PATH = os.path.abspath(appSettings["training_file_name"])
TRAINING_PROMPT = appSettings["training_prompt"]
LOWER_RANGE = int(appSettings["starting_point"])
tokenizer = get_tokenizer("basic_english")
tool = language_tool_python.LanguageTool('en-UK')
FILE_NAME = os.getcwd() + "\\" + appSettings["tone"] + "PromptCompletions.jsonl"
TONE_OF_VOICE = "professional"
API_KEY = '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'
MODEL = "gpt-3.5-turbo"
openai.api_key = API_KEY
os.environ['OPENAI_API_KEY'] = API_KEY

f = open(TRAINING_FILE_PATH, encoding="utf8") #change absolute path to relative path
#print(f.read())
lines = f.readlines()
f.close()

g = open(FILE_NAME, "a") #change absolute path to relative path
trainingJson = ""

for i in range(LOWER_RANGE,len(lines)):
    trainingJson = ""
    if len(lines[i]) > 10:

        try:
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
            #completion.choices[1].content = cleanedPrompt
            cleanedCompletion = tool.correct(completion.choices[0].message.content)
            #print(cleanedCompletion)
            trainingJson += '{"role": "assistant", "content": ' + json.dumps(cleanedCompletion) + '}]}'
            trainingJson +='\n' 
            print(i)
            if i%50==0:
                print(i + " lines parsed")
            g.write(trainingJson)
        
        except:
            print("There may have been some issue with OpenAI's servers. Change the starting_point value to " + i + " and rerun the script")
            break
        
trainingJson = ""    
g.close()
print("Training file successfully created")
