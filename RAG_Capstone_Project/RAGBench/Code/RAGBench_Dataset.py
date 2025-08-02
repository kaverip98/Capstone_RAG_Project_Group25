# Install these first if not installed (uncomment if needed)
!pip install -U datasets sentence-transformers faiss-cpu langchain langchain-community tqdm groq regex gradio chromadb

import os
import zipfile
import json
import regex as re
import torch
from datasets import load_dataset, concatenate_datasets
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from groq import Groq
from google.colab import userdata
from typing import Dict, List, Tuple, Optional
import gradio as gr

# === Global variables ===
stop_initialization = False
vectorstore = None
embeddings = None
groq_model = None
judge_model_settings = None
chunk_settings = None
top_k_settings = None
client = None

# === Prompts ===
RAG_PROMPT_TEMPLATE = """Use the following context to answer:

{context}

Question: {question}

If the answer isn't in the context, say "I don't know"."""

JUDGE_PROMPT_TEMPLATE = """
I asked someone to answer a question based on one or more documents.
Your task is to review their response and assess whether or not each sentence
in that response is supported by text in the documents. And if so, which
sentences in the documents provide that support.

Here are the documents, each split into sentences with keys like '0_0.', '0_1.':
'''
{context}
'''

The question was:
'''
{question}
'''

Here is their response, split into sentences with keys like 'a.', 'b.':
'''
{answer}
'''

You must respond with a JSON object matching this schema:
{{
  "relevance_explanation": string,
  "all_relevant_sentence_keys": [string],
  "overall_supported_explanation": string,
  "overall_supported": boolean,
  "sentence_support_information": [
    {{
      "response_sentence_key": string,
      "explanation": string,
      "supporting_sentence_keys": [string],
      "fully_supported": boolean
    }}
  ],
  "all_utilized_sentence_keys": [string]
}}

⚠️ IMPORTANT:
- Strictly return ONLY the JSON object.
- Do NOT add any text, comments, markdown, or code fences.
- Output must start directly with '{{' and end with '}}'.
"""

# === Helper functions ===
def check_stop():
    global stop_initialization
    if stop_initialization:
        raise Exception("Initialization stopped by user.")

def unzip_if_needed(local_index_path):
    zip_index_path = local_index_path + ".zip"
    if not os.path.exists(local_index_path) and os.path.exists(zip_index_path):
        print(f"Unzipping vectorstore from {zip_index_path}...")
        with zipfile.ZipFile(zip_index_path, 'r') as zip_ref:
            zip_ref.extractall(local_index_path)

def zip_vectorstore(local_index_path):
    zip_index_path = local_index_path + ".zip"
    print(f"Zipping vectorstore to {zip_index_path}...")
    with zipfile.ZipFile(zip_index_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(local_index_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, local_index_path)
                zipf.write(file_path, arcname)

def extract_json_from_text(text):
    matches = re.findall(r'\{(?:[^{}]|(?R))*\}', text, flags=re.DOTALL)
    for m in matches:
        try:
            json.loads(m)
            return m
        except json.JSONDecodeError:
            continue
    return None

def validate_evaluation(data):
    defaults = {
        "relevance_explanation": "N/A",
        "all_relevant_sentence_keys": [],
        "overall_supported_explanation": "N/A",
        "overall_supported": False,
        "sentence_support_information": [],
        "all_utilized_sentence_keys": []
    }
    for k, v in defaults.items():
        if k not in data:
            data[k] = v
    return data

def calculate_metrics(evaluation):
    try:
        relevant = len(evaluation.get("all_relevant_sentence_keys", []))
        utilized = len(evaluation.get("all_utilized_sentence_keys", []))
        supported = sum(1 for s in evaluation.get("sentence_support_information", []) if s.get("fully_supported"))
        total = max(1, len(evaluation.get("sentence_support_information", [])))
        return {
            "context_relevance": relevant / max(1, utilized),
            "context_utilization": utilized / max(1, relevant),
            "completeness": supported / total,
            "adherence": 1.0 if evaluation.get("overall_supported") else 0.0,
            "explanation": f"Relevant:{relevant}, Utilized:{utilized}, Supported:{supported}/{total}"
        }
    except Exception as e:
        print(f"Metric calc error: {e}")
        return {"context_relevance": 0, "context_utilization": 0, "completeness": 0, "adherence": 0, "explanation": "Failed"}

def format_context(docs):
    lines = []
    for i, doc in enumerate(docs):
        for j, s in enumerate(doc["content"].split('. ')):
            if s.strip():
                lines.append(f"{i}_{j}. {s.strip()}")
    return '\n'.join(lines)

# === Groq call ===
def call_groq_model(prompt, model, temperature=0.3, max_tokens=512):
    global client
    if not client:
        return None
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Groq call error: {e}")
        return None

# === Initialize system ===
def initialize_system(database, vectordb, embedding_model, retriever_model, judge_model, chunk_size, chunk_overlap, top_k):
    global vectorstore, embeddings, groq_model, judge_model_settings, chunk_settings, top_k_settings, stop_initialization
    stop_initialization = False

    groq_model = retriever_model
    judge_model_settings = judge_model
    chunk_settings = (chunk_size, chunk_overlap)
    top_k_settings = top_k

    # Auto-select embedding model
    if database == "Medical":
        ragdatasets = ["covidqa", "pubmedqa"]
        embedding_model = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    else:
        if database == "General Knowledge":
          ragdatasets = ["expertqa", "hotpotqa", "msmarco", "hagrid"]
        if database == "Finance":
          ragdatasets = ["finqa", "tatqa"]
        if database == "Customer Support":
          ragdatasets = ["delucionqa", "emanual", "techqa"]
        if database == "Legal":
          ragdatasets = ["cuad"]
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    try:
        datasets_list = []
        for d in ragdatasets:
            check_stop()
            datasets_list.append(load_dataset("rungalileo/ragbench", d, split="test"))
        datasets = concatenate_datasets(datasets_list)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_docs, chunks = [], []
        for row in datasets:
            check_stop()
            for doc in row['documents']:
                if doc.strip():
                    all_docs.append(doc)

        for text in all_docs:
            check_stop()
            chunks.extend(text_splitter.split_text(text))

        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"batch_size": 32, "normalize_embeddings": True}
        )

        local_index_path = f"{vectordb.lower()}_index_{database.lower().replace(' ', '_')}"
        unzip_if_needed(local_index_path)

        if vectordb == "FAISS":
            if os.path.exists(local_index_path):
                vectorstore = FAISS.load_local(local_index_path, embeddings, allow_dangerous_deserialization=True)
            else:
                vectorstore = FAISS.from_texts(chunks, embeddings)
                vectorstore.save_local(local_index_path)
                zip_vectorstore(local_index_path)
        elif vectordb == "Chroma":
            if os.path.exists(local_index_path):
                vectorstore = Chroma(
                    persist_directory=local_index_path,
                    embedding_function=embeddings
                )
            else:
                vectorstore = Chroma.from_texts(
                    texts=chunks,
                    embedding=embeddings,
                    persist_directory=local_index_path
                )
                vectorstore.persist()
                zip_vectorstore(local_index_path)
        else:
            return "Error: Invalid vector database selection", None
    except Exception as e:
        return f"Initialization error: {str(e)}", None

    return f"System initialized with {database}, {vectordb}, embedding: {embedding_model}", f"{len(chunks)} chunks created"

# === Process query ===
def evaluate_answer(question, context, answer):
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, context=context, answer=answer)
    output = call_groq_model(prompt, judge_model_settings, temperature=0.1)
    if not output:
        return None
    json_str = extract_json_from_text(output)
    if not json_str:
        return None
    try:
        data = json.loads(json_str)
        return validate_evaluation(data)
    except Exception as e:
        print(f"JSON parse error: {e}")
        return None

def process_query(question, temperature=0.3, max_tokens=512):
    if not vectorstore:
        return "Error: System not initialized.", None, None, None
    try:
        docs = vectorstore.similarity_search(question, k=top_k_settings)
        formatted = [{"content": d.page_content} for d in docs]
        context = "\n\n".join(d["content"] for d in formatted)

        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        answer = call_groq_model(prompt, groq_model, temperature, max_tokens) or "I don't know"

        judge_context = format_context(formatted)
        eval_result = evaluate_answer(question, judge_context, answer)
        metrics = calculate_metrics(eval_result) if eval_result else {"context_relevance":0,"context_utilization":0,"completeness":0,"adherence":0,"explanation":"Evaluation failed"}

        retrieved_docs = "\n\n".join([f"Document {i+1}:\n{d['content']}" for i, d in enumerate(formatted)])
        eval_output = json.dumps(eval_result, indent=2) if eval_result else "Evaluation failed"
        metrics_output = json.dumps(metrics, indent=2)

        return answer, retrieved_docs, eval_output, metrics_output
    except Exception as e:
        return f"Query error: {str(e)}", None, None, None

# === Gradio interface ===
DATABASE_OPTIONS = ["Medical", "Finance", "General Knowledge", "Customer Support","Legal"]
VECTORDB_OPTIONS = ["FAISS", "Chroma"]
EMBEDDING_MODELS = ["sentence-transformers/all-MiniLM-L6-v2", "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"]
LLM_MODELS = ["llama3-8b-8192", "llama3-70b-8192"]

# --- Groq client ---
groq_api_key = userdata.get('GROQ_API_KEY')
client = Groq(api_key=groq_api_key)

with gr.Blocks(title="RAG System") as demo:
    gr.Markdown("# 🧠 RAG System with Groq for RAGBench Dataset")

    with gr.Tab("Configuration"):
        database = gr.Dropdown(label="Knowledge Domain", choices=DATABASE_OPTIONS, value=DATABASE_OPTIONS[0])
        vectordb = gr.Dropdown(label="Vector Database", choices=VECTORDB_OPTIONS, value=VECTORDB_OPTIONS[0])
        embedding_model = gr.Dropdown(label="Embedding Model", choices=EMBEDDING_MODELS, value=EMBEDDING_MODELS[0])
        retriever_model = gr.Dropdown(label="Retriever LLM", choices=LLM_MODELS, value=LLM_MODELS[0])
        judge_model = gr.Dropdown(label="Judge LLM", choices=LLM_MODELS, value=LLM_MODELS[1])
        chunk_size = gr.Number(label="Chunk Size", value=500)
        chunk_overlap = gr.Number(label="Chunk Overlap", value=100)
        top_k = gr.Number(label="Top K Documents", value=5)
        init_btn = gr.Button("✅ Initilaize Vector DB")
        stop_btn = gr.Button("🛑 Stop Initialization")
        init_output = gr.Textbox(label="Initialization Status", interactive=False)
        chunk_info = gr.Textbox(label="Chunk Information", interactive=False)

    with gr.Tab("Query"):
        question = gr.Textbox(label="Your question")
        temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1.0, step=0.1, value=0.3)
        max_tokens = gr.Slider(label="Max Tokens", minimum=128, maximum=2048, step=64, value=512)
        submit_btn = gr.Button("Submit")

        answer_output = gr.Textbox(label="Answer", interactive=False)
        docs_output = gr.Textbox(label="Retrieved Documents", interactive=False, lines=10)
        eval_output = gr.Textbox(label="Evaluation Results", interactive=False, lines=10)
        metrics_output = gr.Textbox(label="Metrics", interactive=False, lines=5)

    # Events
    def stop_init(): global stop_initialization; stop_initialization=True; return "Stopping initialization..."

    init_btn.click(
        initialize_system,
        inputs=[database, vectordb, embedding_model, retriever_model, judge_model, chunk_size, chunk_overlap, top_k],
        outputs=[init_output, chunk_info]
    )
    stop_btn.click(stop_init, inputs=[], outputs=[init_output])
    submit_btn.click(
        process_query,
        inputs=[question, temperature, max_tokens],
        outputs=[answer_output, docs_output, eval_output, metrics_output]
    )

demo.launch()
