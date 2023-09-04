import json
import openai
import os

with open(os.getcwd() + "\\app.config") as settings_file:
   appSettings = json.load(settings_file)

FILE_NAME = os.getcwd() + "\\" + appSettings["tone"] + "PromptCompletions.jsonl"
API_KEY = appSettings["api_key"]
MODEL = appSettings["model"]
openai.api_key = API_KEY
os.environ['OPENAI_API_KEY'] = API_KEY


trainingFile = openai.File.create(file=open(FILE_NAME, "rb"), purpose='fine-tune')
print("You have to wait a few minutes for OpenAI to process this file")
modelCreated = False

while modelCreated == False:
	userResponse = input("Enter the word 'begin' to send a request to create a fine-tuned model")
	if userResponse == 'begin':
		
		fineTune = openai.FineTuningJob.create(training_file=trainingFile.id, model = "gpt-3.5-turbo",hyperparameters={"n_epochs":1,"suffix":appSettings["tone"]})
		print("Fine tuning process started")
		