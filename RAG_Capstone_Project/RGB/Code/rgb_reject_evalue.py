import json
import tqdm, os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def check(question, answer, model):
    prompt = '''I will give you a question and an answer generated through document retrieval. Please use this answer to determine if the retrieved document can solve the question.
Demonstrations:
Question: 2023年澳网女单冠军是谁
Answer:文档信息不足，因此我无法基于提供的文档回答该问题。
No, the question is not addressed by the documents.

Question: Who is the champion of Australian Open 2023 Women's Singles?
Answer: Serena Williams
Yes, the question is addressed by the documents.

Question: Where is ACL2023 held?
Answer: Location of ACL2023 has not been confirmed.
No, the question is not addressed by the documents.

Question:  2023年中国GDP是多少?
Answer: I can not answer this question。
No, the question is not addressed by the documents.

Begin to generate:
Question: {question}
Answer: {answer}
    '''
    text2 = prompt.format(question=question,answer=answer)
    return getdata(text2,model)

def getdata(text,model):
    messages=[]
    messages.append([{"role": "user", "content": text}])
    print("AT ModelGenerate")
    prediction = model.generate(messages)
    completion = prediction
    return completion

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
    # Ensure pad_token_id is set
      if self.tokenizer.pad_token_id is None:
          self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
      if self.model.config.pad_token_id is None:
          self.model.config.pad_token_id = self.model.config.eos_token_id


    def generate(self,messages):
      model_inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
      generated_ids = self.model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
      decoded = self.tokenizer.batch_decode(generated_ids)
      return decoded[0]

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ChatglmModel:
    def __init__(self, plm) -> None:
        model_name = plm

        # Load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        # Ensure pad_token_id is set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def generate(self, messages):
        # Tokenize with attention mask
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,  # Ensure attention_mask is created
            add_special_tokens=True
        )

        # Move inputs to model device
        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}

        # Generate
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=1000,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id  # Explicit pad_token_id
        )

        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded[0]

class ChatglmModel:
    def __init__(self, plm) -> None:
        model_name = plm

        # Set device to CPU
        self.device = torch.device("cpu")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,   # Use float32 for CPU
            device_map=None,             # Avoid device_map="auto"
            trust_remote_code=True
        ).to(self.device)                # Move model to CPU

        # Set pad_token_id to eos_token_id if not set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def generate(self, messages):
        # Tokenize with attention mask and move to CPU
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True
        )
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        # Generate output
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=1000,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )

        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded[0]

from google.colab import userdata
from huggingface_hub import login

# Paste your token below

login(userdata.get('HF_TOKEN'))

modelname = "mini-llama-2-1.1b"
temp = 0.2
noise_rate = 1.0
passage_num = 5
dataset = 'en_fact'
factchecking = False
correct_rate = 0.0
plm = 'Qwen/Qwen2.5-VL-3B-Instruct'
plm2 = 'mistralai/Mistral-7B-Instruct-v0.2'
url = ''
api_key = ''
instances = []

if 'en' in dataset:
        resultpath = 'result-en'

model = ChatglmModel(plm2)

import re
def get_substring(text):
  match = re.search(r'\[/INST\](.*?)</s>', text)
  substring = match.group(1)
  return substring

evaluefile = f'prediction_{dataset}_{modelname}_temp{temp}_noise{1.0}_passage{passage_num}_correct{0.0}.json'

outputfile = f'prediction_{dataset}_{modelname}_temp{temp}_noise{1.0}_passage{passage_num}_correct{0.0}_chatgpt.json'

resultfile = f'prediction_{dataset}_{modelname}_temp{temp}_noise{1.0}_passage{passage_num}_correct{0.0}_chatgptresult.json'



results = []
useddata = {}
if os.path.exists(outputfile):
    with open(outputfile) as f:
        for line in f:
            data = json.loads(line)
            useddata[data['id']] = data

print(len(useddata))

i=0
with open(outputfile,'w',encoding='utf-8') as f:
    with open(evaluefile, 'r', encoding='utf-8') as f2:
        for line in tqdm.tqdm(f2):
            print(i+1)
            data = json.loads(line)
            if data['id'] in useddata and data['query'] == useddata[data['id']]['query'] and data['ans']  == useddata[data['id']]['ans'] :
                results.append(useddata[data['id']])
                f.write(json.dumps(useddata[data['id']],ensure_ascii=False)+'\n')
                print("written")
                continue
            try:
                question = data['query']
                answer = data['prediction']
                # print("Answer",answer)
                evaluation_ = check(question, answer, model)

                evaluation = get_substring(evaluation_)
                data['evaluation'] = evaluation
                results.append(data)
                f.write(json.dumps(data,ensure_ascii=False)+'\n')
            except Exception as e:
                print(e)
                print(question,answer)
                break
                # print(question,answer)
                continue

rejecttt = 0
tt = 0
for i in results:
    # print(type(i))
    # print(i)
    if "not addressed" in i['evaluation']:
        rejecttt += 1
    if 0 not in i['label'] and 1 in i['label']:
        tt += 1
print(tt/len(results))
scores = {
    'reject_rate': rejecttt/len(results),
    'all_rate': (tt)/len(results),
    'tt':tt,
    'rejecttt':rejecttt,
    'nums': len(results),
}
json.dump(scores, open(resultfile, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
