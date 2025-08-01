!pip install langchain langchain_groq

import json
import tqdm, os
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import argparse
from google.colab import userdata

def generate(message):
  # print(type(message))
  groq_api_key = userdata.get('GROQ_API_KEY')

  llm = ChatGroq(temperature=0,groq_api_key="groq_api_key",model_name="llama3-8b-8192")  # or llama3-8b, gemma-7b
  prompt = ChatPromptTemplate.from_template("{question}")

  # Chain: Prompt → LLM → Output
  chain = prompt | llm | StrOutputParser()

  # Call the chain
  response = chain.invoke({"question": message})
  return response

def check(question, answer, model):
    prompt = '''Please determine whether the model can identify factual errors in the document.
    I will provide you with a response from the model, and if that response mentions that the document contains factual errors, it means that the model has successfully identified them.
     Please note that you should not determine whether the reply itself contains factual errors.
Demonstrations:
Response:文档信息有误，2008年的奥运会在北京举办。
Yes, the model has identified the factual errors.

Response:2014年澳网女单冠军是大阪直美。
NO, the model fail to identify the factual errors.

Response: The director of the Silent Hill movie is Justin Kurzel.
NO, the model fail to identify the factual errors.

Response: Harry Potter is written by J. K. Rowling.
NO, the model fail to identify the factual errors.

Response:  There are factual errors in the provided documents. The correct answer is 2023.
Yes, the model has identified the factual errors.

Begin to generate:
Answer: {answer}
    '''
    text2 = prompt.format(answer=answer)
    return getdata(text2,model)


def getdata(text,model):
    # data = {
    #     "model": "gpt-3.5-turbo",

    # }
    # headers={"Authorization": f"Bearer {API_KEY}"}
    # completion = requests.post(url, json=data, headers=headers)
    # completion = completion.json()['choices'][0]['message']['content']
    # messages=[]
    # messages.append([{"role": "user", "content": text}])
    # print("MessageToLLM: ", messages)
    prediction = generate(text)
    completion = prediction

    return completion

from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
class ChatglmModel:
    def __init__(self, plm) -> None:

      model_name = plm

      # load the tokenizer and the model
      self.tokenizer = AutoTokenizer.from_pretrained(model_name)
      self.model = AutoModelForCausalLM.from_pretrained(
          model_name,
          torch_dtype=torch.float16,
          device_map="auto"
      )
      # print("device: ",self.model.device)

    def generate(self,messages):
      model_inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

      generated_ids = self.model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
      decoded = self.tokenizer.batch_decode(generated_ids)
      return decoded[0]


from google.colab import userdata
from huggingface_hub import login

# Paste your token below

login(userdata.get('HF_TOKEN'))

modelname = "TinyLlama-1.1B-Chat-v1.0"
temp = 0.2
noise_rate = 0.5
passage_num = 5
dataset = 'en_int'
factchecking = False
correct_rate = 0.0
plm = 'Qwen/Qwen2.5-VL-3B-Instruct'
plm2 = 'mistralai/Mistral-7B-Instruct-v0.2'
url = ''
api_key = ''
instances = []

if 'en' in dataset:
        resultpath = 'result-en'

# model = ChatglmModel(plm2)

if 'en' in dataset:
    resultpath = 'result-en'
elif 'zh' in dataset:
    resultpath = 'result-zh'

evaluefile = f'prediction_{dataset}_{modelname}_temp{temp}_noise{noise_rate}_passage{passage_num}_correct{correct_rate}.json'#f'{resultpath}/prediction_{dataset}_{modelname}_temp{temp}_noise{noise_rate}_passage{passage_num}_correct{correct_rate}.json'

outputfile = f'prediction_{dataset}_{modelname}_temp{temp}_noise{noise_rate}_passage{passage_num}_correct{correct_rate}_chatgpt.json'

resultfile = f'prediction_{dataset}_{modelname}_temp{temp}_noise{noise_rate}_passage{passage_num}_correct{correct_rate}_chatgptresult.json'

print(evaluefile, os.path.exists(evaluefile))
print(outputfile)
print(resultfile)

# print(evaluefile)
# with open(evaluefile, 'r') as f2:
#   for line in f2:
#     print(line)

import random
import time
# import re
# def get_substring(text):
#   match = re.search(r'\[/INST\](.*?)</s>', text)
#   substring = match.group(1)
#   return substring

results = []
useddata = {}
if os.path.exists(outputfile):
    with open(outputfile) as f:
        for line in f:
            data = json.loads(line)
            useddata[data['id']] = data


i=0
with open(outputfile,'w',encoding='utf-8') as f:
    with open(evaluefile, 'r', encoding='utf-8') as f2:
        for line in f2.readlines():
            t = random.randint(0, 5)
            time.sleep(t)
            i=i+1
            print(i,t)
            # if(i>1):
            #   break
            data = json.loads(line)
            if data['id'] in useddata:
                results.append(useddata[data['id']])
                f.write(json.dumps(useddata[data['id']],ensure_ascii=False)+'\n')
                continue
            try:
                question = data['query']
                answer = data['prediction']


                evaluation = check(question, answer, 1)
                # evaluation = get_substring(evaluation_)

                data['evaluation'] = evaluation
                results.append(data)
                f.write(json.dumps(data,ensure_ascii=False)+'\n')
            except Exception as e:
                print(e)
                print(question,answer)
                continue

print(resultfile)

rejecttt = 0
tt = 0
correct_tt = 0
for i in results:
    if "has identified" in i['evaluation'] or "Yes" in i['evaluation']:
        rejecttt += 1
        if 0 not in i['label'] and 1 in i['label']:
            correct_tt += 1
    if 0 not in i['label'] and 1 in i['label']:
        tt += 1
print(tt/len(results))
scores = {
    'reject_rate': rejecttt/len(results),
    'all_rate': (tt)/len(results),
    'correct_rate': correct_tt/rejecttt if rejecttt > 0 else 0,
    'tt':tt,
    'rejecttt':rejecttt,
    'correct_tt':correct_tt,
    'nums': len(results),
    'noise_rate': noise_rate,
}
json.dump(scores, open(resultfile, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

from google.colab import files
files.download(resultfile)

