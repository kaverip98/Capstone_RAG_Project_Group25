# RAG_Capstone_Project_Group25

Objective:
Develop an efficient Retrieval-Augmented Generation (RAG) pipeline capable of delivering
accurate answers to the user queries.

Dataset : https://huggingface.co/datasets/rungalileo/ragbench

Dataset Description : RAG Bench is a large-scale benchmark dataset specifically designed for
training and evaluating RAG systems. It contains 100k examples spanning five industry-specific
domains:

● Biomedical Research
● General Knowledge
● Legal
● Customer Support
● Finance

The dataset supports various RAG task types and includes evaluation metrics such as context
relevance, context utilization, answer faithfulness and answer completeness.
The dataset samples are sourced from real-world industry corpora, such as user manuals, making
the dataset highly relevant for practical applications. This comprehensive dataset enables a
detailed assessment of RAG system performance, ensuring robust and reliable evaluation.

Project Overview: Generative large language models often face challenges such as producing
outdated information or fabricating facts. Retrieval-Augmented Generation (RAG) techniques
address these limitations by integrating pretraining and retrieval-based approaches. This
combination creates a robust framework for improving model accuracy and reliability.

RAG offers the added benefit of enabling rapid deployment of domain-specific applications
without requiring updates to the model parameters. As long as relevant documents are available
for a query, the system can adapt to organizational or domain-specific needs efficiently.

A typical RAG workflow involves Query Classification, Retrieval, Reranking, Repacking and
Summarization. Implementing RAG requires careful consideration of various factors, such as
properly splitting documents into manageable chunks, selecting suitable embeddings to
semantically represent these chunks, choosing vector databases for efficient storage and retrieval
of feature representations.

By addressing these elements, RAG systems can deliver accurate, domain-specific, and
context-aware responses, making them a powerful tool for real-world applications.
References:
1) RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems
2) Searching for Best Practices in Retrieval-Augmented Generation,

Tools: Transformers, Hugging Face, PyTorch, Langchain, LlamaIndex
