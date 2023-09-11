import json
import openai
import os

#loads the config file
with open(os.getcwd() + "\\app.config") as settings_file:
   appSettings = json.load(settings_file)

#defines constants based off of values from app.config
FILE_NAME = os.getcwd() + "\\" + appSettings["tone"] + "PromptCompletions.jsonl"
API_KEY = appSettings["api_key"]
MODEL = appSettings["model"]
SUFFIX = appSettings["tone"]
openai.api_key = API_KEY
os.environ['OPENAI_API_KEY'] = API_KEY


trainingFile = openai.File.create(file=open(FILE_NAME, "rb"), purpose='fine-tune')
print("You have to wait a few minutes for OpenAI to process this file")
modelCreated = False

while modelCreated == False:
    userResponse = input("Enter the word 'begin' to send a request to create a fine-tuned model: ")
    if userResponse == 'begin':
        try:
            print("Fine tuning process started")
            #1 epoch is generally enough, and is the cheapest option by far. Having multiple models with the same suffix is fine.
            fineTune = openai.FineTuningJob.create(training_file=trainingFile.id, model = MODEL,hyperparameters={"n_epochs":1}, suffix=SUFFIX)
			modelCreated = True
        except:
            print("OpenAI hasn't finished processing the file") #this takes a few minutes
