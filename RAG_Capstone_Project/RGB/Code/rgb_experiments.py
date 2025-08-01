
!pip install langchain langchain_groq

import json
import numpy as np
import random, math
import argparse,torch
import os
import json, tqdm, requests
import yaml
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

def predict_groq(query, ground_truth, docs, model, system, instruction, temperature, dataset):

    '''
    label: 0 for positive, 1 for negative, -1 for not enough information

    '''
    if len(docs) == 0:

        text = instruction.format(QUERY=query, DOCS='')
        # if len(system) > 0:
        #     messages.append({"role": "system", "content": system})
        # messages.append({"role": "user", "content": text})
        print("here2")
        messages = f"role": "user", "content": text"
        prediction = generate(messages)

    else:

        docs = '\n'.join(docs)
        messages=[]
        text = instruction.format(QUERY=query, DOCS=docs)
        # if len(system) > 0:
        #     messages.append({"role": "system", "content": system})
        # messages.append({"role": "user", "content": text})
        messages = f"role": "user", "content": text"


        print("here3")
        prediction = generate(messages)

    if 'zh' in dataset:
        prediction = prediction.replace(" ","")

    if '信息不足' in prediction or 'insufficient information' in prediction:
        labels = [-1]
    else:
        labels = checkanswer(prediction, ground_truth)

    factlabel = 0

    if '事实性错误' in prediction or 'factual errors' in prediction:
        factlabel = 1

    return labels,prediction, factlabel

def processdata(instance, noise_rate, passage_num, filename, correct_rate = 0):
    query = instance['query']
    ans = instance['answer']

    neg_num = math.ceil(passage_num * noise_rate)
    pos_num = passage_num - neg_num

    if '_int' in filename:
        for i in instance['positive']:
            random.shuffle(i)
        print(len(instance['positive']))
        docs = [i[0] for i in instance['positive']]
        if len(docs) < pos_num:
            maxnum = max([len(i) for i in instance['positive']])
            for i in range(1,maxnum):
                for j in instance['positive']:
                    if len(j) > i:
                        docs.append(j[i])
                        if len(docs) == pos_num:
                            break
                if len(docs) == pos_num:
                    break
        neg_num = passage_num - len(docs)
        if neg_num > 0:
            negative = instance['negative'][:neg_num]
            docs += negative
    elif '_fact' in filename:
        correct_num = math.ceil(passage_num * correct_rate)
        pos_num = passage_num - neg_num - correct_num
        indexs = list(range(len(instance['positive'])))
        selected = random.sample(indexs,min(len(indexs),pos_num))
        docs = [instance['positive_wrong'][i] for i in selected]
        remain = [i for i in indexs if i not in selected]
        if correct_num > 0 and len(remain) > 0:
            docs += [instance['positive'][i] for i in random.sample(remain,min(len(remain),correct_num))]
        if neg_num > 0:
            docs += instance['negative'][:neg_num]
    else:
        if noise_rate == 1:
            neg_num = passage_num
            pos_num = 0
        else:
            if neg_num > len(instance['negative']):
                neg_num = len(instance['negative'])
                pos_num = passage_num - neg_num
            elif pos_num > len(instance['positive']):
                pos_num = len(instance['positive'])
                neg_num = passage_num - pos_num


        positive = instance['positive'][:pos_num]
        negative = instance['negative'][:neg_num]

        docs = positive + negative

    random.shuffle(docs)

    return query, ans, docs

def checkanswer(prediction, ground_truth):
    prediction = prediction.lower()
    if type(ground_truth) is not list:
        ground_truth = [ground_truth]
    labels = []
    for instance in ground_truth:
        flag = True
        if type(instance)  == list:
            flag = False
            instance = [i.lower() for i in instance]
            for i in instance:
                if i in prediction:
                    flag = True
                    break
        else:
            instance = instance.lower()
            if instance not in prediction:
                flag = False
        labels.append(int(flag))
    return labels

def getevalue(results):
    results = np.array(results)
    results = np.max(results,axis = 0)
    if 0 in results:
        return False
    else:
        return True

def predict(query, ground_truth, docs, model, system, instruction, temperature, dataset):

    '''
    label: 0 for positive, 1 for negative, -1 for not enough information

    '''
    if len(docs) == 0:

        text = instruction.format(QUERY=query, DOCS='')
        if len(system) > 0:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": text})
        print("here2")
        prediction = model.generate(messages)

    else:

        docs = '\n'.join(docs)
        messages=[]
        text = instruction.format(QUERY=query, DOCS=docs)
        if len(system) > 0:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": text})


        print("here3")
        prediction = model.generate(messages)

    if 'zh' in dataset:
        prediction = prediction.replace(" ","")

    if '信息不足' in prediction or 'insufficient information' in prediction:
        labels = [-1]
    else:
        labels = checkanswer(prediction, ground_truth)

    factlabel = 0

    if '事实性错误' in prediction or 'factual errors' in prediction:
        factlabel = 1

    return labels,prediction, factlabel

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
class TinyLlama:
    def __init__(self, plm) -> None:

      model_name = plm

      # load the tokenizer and the model
      self.tokenizer = AutoTokenizer.from_pretrained(model_name)
      self.model = AutoModelForCausalLM.from_pretrained(
          model_name,
          torch_dtype=torch.float16,
          device_map="auto"
      )
      print("device: ",self.model.device)

    def generate(self,messages):
      text = self.tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True,
          enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
      )
      model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

      # conduct text completion
      generated_ids = self.model.generate(
          **model_inputs,
          max_new_tokens=32768
      )
      output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

      # parsing thinking content
      try:
          # rindex finding 151668 (</think>)
          index = len(output_ids) - output_ids[::-1].index(151668)
      except ValueError:
          index = 0

      thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
      content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
      print("content:", content)
      return content

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLMModel:
    def __init__(self, model_name: str, device: str = None, use_chat_template: bool = False):
        """
        Initialize the LLM from Hugging Face.

        Args:
            model_name (str): Hugging Face model ID
            device (str): 'cuda', 'cpu', or None (auto-detect)
            use_chat_template (bool): Apply chat formatting if available
        """
        self.model_name = model_name
        self.use_chat_template = use_chat_template

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Move to device
        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()

        # Check if chat template exists
        self.has_chat_template = hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None

    def generate(self, messages):
        """
        Generate text using the model.

        Args:
            prompt (str): Input text or chat prompt
            max_new_tokens (int): Number of tokens to generate
        """
        max_new_tokens = 200
        if self.use_chat_template and self.has_chat_template:
            # For chat models with a defined template
            # messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", tokenize=True
            )

            # Build attention mask manually since we used apply_chat_template
            inputs = {
                "input_ids": input_ids.to(self.device),
                "attention_mask": torch.ones_like(input_ids).to(self.device)
            }
        else:
            # For regular models
            tokens = self.tokenizer(prompt, return_tensors="pt")
            inputs = {
                "input_ids": tokens["input_ids"].to(self.device),
                "attention_mask": tokens["attention_mask"].to(self.device)
            }

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text

prompt = {'system': "You are an accurate and reliable AI assistant that can answer questions with the help of external documents. Please note that external documents may contain noisy or factually incorrect information. If the information in the document contains the correct answer, you will give an accurate answer. If the information in the document does not contain the answer, you will generate ’I can not answer the question because of the insufficient information in documents.‘. If there are inconsistencies with the facts in some of the documents, please generate the response 'There are factual errors in the provided documents.' and provide the correct answer.",
          'instruction': "Document:\n{DOCS} \n\nQuestion:\n{QUERY}"}

system = prompt['system']
instruction = prompt['instruction']

#change the modelname, run for noise rates 0.0, 0.25, 0.50, 1.0
#run for datasets en_fact and en_refine and en_init

modelname = "GROQ"
temperature = 0.2
noise_rate = 0.5
passage_num = 5
dataset = "en_int"
factchecking = False
correct_rate = 0.0
plm = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
url = ''
api_key = ''
instances = []

from huggingface_hub import login

# Paste your token below
from google.colab import userdata
from huggingface_hub import login

login(userdata.get('HF_TOKEN'))

model2 = TinyLlama(plm)
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# llm2 = LLMModel(model_id, use_chat_template=True)

with open(f'/content/{dataset}.json','r',  encoding="utf-8") as f:
    for line in f:
        instances.append(json.loads(line))
if 'en' in dataset:
    resultpath = 'result-en'
elif 'zh' in dataset:
    resultpath = 'result-zh'
if not os.path.exists(resultpath):
    os.mkdir(resultpath)

filename = f'{resultpath}/prediction_{dataset}_{modelname}_temp{temperature}_noise{noise_rate}_passage{passage_num}_correct{correct_rate}.json'
useddata = {}
if os.path.exists(filename):
    with open(filename) as f:
        for line in f:
            data = json.loads(line)
            useddata[data['id']] = data

len(useddata)

results = []
i=0
with open(filename,'w') as f:
    for instance in tqdm.tqdm(instances):
      i=i+1
      if i>=50:
        break
      if instance['id'] in useddata and instance['query'] == useddata[instance['id']]['query'] and instance['answer']  == useddata[instance['id']]['ans']:
          results.append(useddata[instance['id']])
          f.write(json.dumps(useddata[instance['id']], ensure_ascii=False)+'\n')
          continue
      try:
          random.seed(2333)
          if passage_num == 0:
              query = instance['query']
              ans = instance['answer']
              docs = []
          else:
              query, ans, docs = processdata(instance, noise_rate, passage_num, dataset, correct_rate)

          print("here")
          label,prediction,factlabel = predict_groq(query, ans, docs, model2,system,instruction,temperature,dataset)
          instance['label'] = label
          newinstance = {
              'id': instance['id'],
              'query': query,
              'ans': ans,
              'label': label,
              'prediction': prediction,
              'docs': docs,
              'noise_rate': noise_rate,
              'factlabel': factlabel
          }
          results.append(newinstance)
          f.write(json.dumps(newinstance, ensure_ascii=False)+'\n')

      except Exception as e:
          print("Error:", e)
          continue

"""**Experimenting with LLM on GPU**"""

from google.colab import files
files.download(resultpath)  
