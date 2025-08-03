!pip install langchain-community langchain-text-splitters faiss-cpu "datasets>=2.17.0" "groq>=0.4.0" "gradio>=4.20.0" "scikit-learn>=1.4.0" regex chromadb

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
from sklearn.metrics import mean_squared_error, roc_auc_score
import numpy as np

# === Global variables ===
stop_initialization = False
vectorstore = None
embeddings = None
groq_model = None
judge_model_settings = None
chunk_settings = None
top_k_settings = None
client = None
predicted_metrics = []
ground_truth_labels = []

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
            "context_relevance": relevant / max(1, utilized) if utilized > 0 else 0,
            "context_utilization": utilized / max(1, relevant) if relevant > 0 else 0,
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

        if os.path.exists(local_index_path):
            if vectordb == "FAISS":
                vectorstore = FAISS.load_local(local_index_path, embeddings, allow_dangerous_deserialization=True)
            elif vectordb == "Chroma":
                vectorstore = Chroma(
                    persist_directory=local_index_path,
                    embedding_function=embeddings
                )
        else:
            if vectordb == "FAISS":
                vectorstore = FAISS.from_texts(chunks, embeddings)
                vectorstore.save_local(local_index_path)
                zip_vectorstore(local_index_path)
            elif vectordb == "Chroma":
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

ragdatasets = ['covidqa', 'cuad', 'delucionqa', 'emanual', 'expertqa', 'finqa', 'hagrid', 'hotpotqa', 'msmarco', 'pubmedqa', 'tatqa', 'techqa']

for d in ragdatasets:
    ds = load_dataset("rungalileo/ragbench", d, split="test")
    for item in ds:
        ground_truth_labels.append({
            "question": item["question"],
            "context_relevance": item.get("context_relevance_score", 0),
            "context_utilization": item.get("context_utilization_score", 0),
            "completeness": item.get("completeness_score", 0),
            "adherence": item.get("adherence_score", 0)
        })

def compute_rmse_auc(metrics_data):
    if not metrics_data:
        return "No predictions yet."

    y_true = {"context_relevance": [], "context_utilization": [], "completeness": [], "adherence": []}
    y_pred = {"context_relevance": [], "context_utilization": [], "completeness": [], "adherence": []}

    for entry in metrics_data:
        if "true" not in entry or "pred" not in entry:
            continue
        for key in y_true:
            if key in entry["true"] and key in entry["pred"]:
                y_true[key].append(entry["true"][key])
                y_pred[key].append(entry["pred"][key])

    if not any(y_true.values()):
        return "No valid metric data found to process."

    results = {}
    for key in y_true:
        if not y_true[key]:
            continue
        rmse = np.sqrt(mean_squared_error(y_true[key], y_pred[key]))
        try:
            if key == "adherence" and len(np.unique(y_true[key])) > 1:
                auc = roc_auc_score(y_true[key], y_pred[key])
            else:
                auc = "N/A"
        except Exception as e:
            auc = f"Error: {e}"
        results[key] = {"RMSE": round(rmse, 4), "AUC-ROC": auc}

    return json.dumps(results, indent=2)

DATABASE_OPTIONS = ["Medical", "Finance", "General Knowledge", "Customer Support","Legal"]
VECTORDB_OPTIONS = ["FAISS", "Chroma"]
EMBEDDING_MODELS = ["sentence-transformers/all-MiniLM-L6-v2", "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"]
LLM_MODELS = ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]

groq_api_key = userdata.get('GROQ_API_KEY')
client = Groq(api_key=groq_api_key)

with gr.Blocks(title="RAG System") as demo:
    gr.Markdown("# 🧠 RAG System with Groq")

    predicted_metrics_state = gr.State([])

    with gr.Tab("Configuration"):
        database = gr.Dropdown(label="Knowledge Domain", choices=DATABASE_OPTIONS, value=DATABASE_OPTIONS[0])
        vectordb = gr.Dropdown(label="Vector Database", choices=VECTORDB_OPTIONS, value=VECTORDB_OPTIONS[0])
        embedding_model = gr.Dropdown(label="Embedding Model", choices=EMBEDDING_MODELS, value=EMBEDDING_MODELS[0], interactive=True)
        retriever_model = gr.Dropdown(label="Retriever LLM", choices=LLM_MODELS, value=LLM_MODELS[0])
        judge_model = gr.Dropdown(label="Judge LLM", choices=LLM_MODELS, value=LLM_MODELS[1])
        chunk_size = gr.Number(label="Chunk Size", value=500)
        chunk_overlap = gr.Number(label="Chunk Overlap", value=100)
        top_k = gr.Number(label="Top K Documents", value=5)
        init_btn = gr.Button("✅ Initialize System")
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

    with gr.Tab("Evaluation"):
        evaluate_all_btn = gr.Button("📊 Evaluate RMSE & AUC")
        rmse_output = gr.Textbox(label="RMSE and AUC-ROC Results", interactive=False)
        view_data_btn = gr.Button("View Collected Data")
        data_output = gr.JSON(label="Collected Metrics Data")

    def stop_init():
        global stop_initialization
        stop_initialization = True
        return "Stopping initialization..."

    def process_and_update_metrics(question, temperature, max_tokens, current_metrics_list):
        global predicted_metrics
        predicted_metrics = []

        answer, retrieved_docs, eval_output, metrics_output_str = process_query(question, temperature, max_tokens)

        if predicted_metrics:
            new_metric = predicted_metrics[-1]
            updated_list = current_metrics_list + [new_metric]
        else:
            updated_list = current_metrics_list

        return answer, retrieved_docs, eval_output, metrics_output_str, updated_list

    init_btn.click(
        initialize_system,
        inputs=[database, vectordb, embedding_model, retriever_model, judge_model, chunk_size, chunk_overlap, top_k],
        outputs=[init_output, chunk_info]
    )
    stop_btn.click(stop_init, inputs=[], outputs=[init_output])

    submit_btn.click(
        fn=process_and_update_metrics,
        inputs=[question, temperature, max_tokens, predicted_metrics_state],
        outputs=[answer_output, docs_output, eval_output, metrics_output, predicted_metrics_state]
    )

    evaluate_all_btn.click(
        fn=compute_rmse_auc,
        inputs=[predicted_metrics_state],
        outputs=[rmse_output]
    )

    view_data_btn.click(lambda x: x, inputs=[predicted_metrics_state], outputs=[data_output])

"""## 📊 Evaluation Metrics Visualization

The following plots provide a visual representation of the RAG system's performance based on the queries you have run through the interface.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# This function will be called from the Gradio interface
def plot_metrics(metrics_data):
    if not metrics_data:
        print("No data available to plot. Please run some queries first.")
        return None, None

    # 1. Prepare data for plotting
    y_true = {"context_relevance": [], "context_utilization": [], "completeness": [], "adherence": []}
    y_pred = {"context_relevance": [], "context_utilization": [], "completeness": [], "adherence": []}

    for entry in metrics_data:
        for key in y_true:
            y_true[key].append(entry["true"][key])
            y_pred[key].append(entry["pred"][key])

    # 2. Bar Chart for RMSE Scores
    rmse_scores = {}
    for key in y_true:
        if y_true[key]:
            rmse = np.sqrt(mean_squared_error(y_true[key], y_pred[key]))
            rmse_scores[key] = rmse

    if not rmse_scores:
        print("Not enough data to calculate RMSE.")
        return None, None

    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    metrics_names = list(rmse_scores.keys())
    scores = list(rmse_scores.values())
    sns.barplot(x=metrics_names, y=scores, ax=ax1, palette="viridis")
    ax1.set_title('RMSE Scores for RAG Metrics', fontsize=16)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_xlabel('Metric', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 3. Histograms for Metric Distributions
    num_metrics = len(y_true.keys())
    fig2, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    fig2.suptitle('Distribution of True vs. Predicted Scores', fontsize=20)

    for i, key in enumerate(y_true.keys()):
        if y_true[key]:
            sns.histplot(y_true[key], color="skyblue", label='True Values', ax=axes[i], kde=True, stat="density", linewidth=0)
            sns.histplot(y_pred[key], color="red", label='Predicted Values', ax=axes[i], kde=True, stat="density", linewidth=0, alpha=0.6)
            axes[i].set_title(f'{key.replace("_", " ").title()} Distribution')
            axes[i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig1, fig2

# Add a new tab to Gradio for plots
with demo:
    with gr.Tab("Visualization"):
        plot_btn = gr.Button("📈 Generate Plots")
        with gr.Row():
            plot_output_rmse = gr.Plot(label="RMSE Scores Bar Chart")
            plot_output_dist = gr.Plot(label="Score Distributions")

    plot_btn.click(
        fn=plot_metrics,
        inputs=[predicted_metrics_state],
        outputs=[plot_output_rmse, plot_output_dist]
    )

demo.launch(share=True, debug=False)



import matplotlib.pyplot as plt
import numpy as np

# Data
datasets = ['FINANCE', 'LEGAL', 'GENERAL KNOWLEDGE', 'CUSTOMER SUPPORT', 'BIOMEDICAL RESEARCH']
rmse = [0.11, 0.2, 0.15, 0.12, 0.14]
aucroc = [0.72, 0.76, 0.89, 0.91, 0.87]

# Bar width and positions
x = np.arange(len(datasets))
width = 0.35

# Plot
plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - width/2, rmse, width, label='RMSE', color='skyblue')
bars2 = plt.bar(x + width/2, aucroc, width, label='AUCROC', color='salmon')

# Labels and Title
plt.ylabel('Scores')
plt.title('RMSE and AUCROC Across Datasets')
plt.xticks(x, datasets, rotation=45, ha='right')
plt.legend()

# Annotate values on bars
for bar in bars1 + bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

