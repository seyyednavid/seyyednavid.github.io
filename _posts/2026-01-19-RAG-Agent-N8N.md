---
layout: post
title: iOS 18 Documentation Chatbot Using RAG (n8n + Pinecone + OpenAI)
image: "/posts/ios18-rag-agent-title.png"
tags: [GenAI, RAG, LLMs, n8n, Pinecone, OpenAI]
---

In this project, I built an end-to-end **Retrieval-Augmented Generation (RAG)** system that allows users to ask natural-language questions about **iOS 18 features**, with answers grounded strictly in official Apple documentation.

The system uses **n8n** for orchestration, **Pinecone** for vector storage, and **OpenAI** for embeddings and response generation, and is deployed with a **public chat interface**.

---

# Table of Contents

- [00. Project Overview](#overview-main)
  - [Context](#overview-context)
  - [Actions](#overview-actions)
  - [Results](#overview-results)
  - [Growth / Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
- [02. RAG Architecture](#rag-architecture)
- [03. Document Ingestion Pipeline](#rag-ingestion)
- [04. Query & Retrieval Pipeline](#rag-query)
- [05. Application & Example Queries](#rag-application)
- [06. Growth & Next Steps](#growth-next-steps)

___

# 00. Project Overview <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Large Language Models are powerful, but their knowledge is **static** and prone to hallucination when answering questions about newly released or domain-specific content.

With the release of **iOS 18**, users need a reliable way to query official feature documentation without relying on incomplete or outdated model knowledge.

The goal of this project was to build a **document-grounded AI agent** that answers questions about iOS 18 *only* using verified source material.

---

### Actions <a name="overview-actions"></a>

I designed and implemented a full RAG pipeline that:

- Ingests official iOS 18 documentation (PDF)
- Splits content into semantically meaningful chunks
- Generates dense vector embeddings
- Stores embeddings in Pinecone
- Retrieves only relevant chunks at query time
- Generates grounded answers via an AI agent
- Exposes a public chat interface using n8n

All orchestration, retrieval, and generation logic is handled visually and programmatically inside **n8n workflows**.

---

### Results <a name="overview-results"></a>

The final system:

- Answers questions **strictly based on the source document**
- Prevents hallucinations by rejecting unsupported queries
- Returns concise, accurate feature explanations
- Supports public access via a hosted chat UI
- Demonstrates a production-style RAG architecture without custom backend code

---

### Growth / Next Steps <a name="overview-growth"></a>

Potential future enhancements include:

- Source citation per response
- Multi-document ingestion and namespace separation
- Authentication and rate limiting for public chat
- Streaming responses for improved UX
- Automatic document re-indexing on file updates

___

# 01. Data Overview <a name="data-overview"></a>

The knowledge base consists of an official Apple PDF:

- **iOS 18 – All New Features (September 2024)**

This document serves as the *single source of truth* for the system.

It is ingested once, chunked, embedded, and persisted in a vector database, allowing efficient semantic retrieval at query time.

___

# 02. RAG Architecture <a name="rag-architecture"></a>

The system follows a classic Retrieval-Augmented Generation architecture:

![Architecture]("/posts/architecture.jpg")

At a high level:

1. Documents are ingested and vectorised
2. User queries are embedded
3. Relevant chunks are retrieved from Pinecone
4. Retrieved context is injected into the AI agent prompt
5. The model generates a grounded answer

___

# 03. Document Ingestion Pipeline <a name="rag-ingestion"></a>

![Ingestion Flow]("/posts/ingestion-flow.jpg")

The ingestion workflow includes:

- PDF loading
- Recursive text chunking
- Embedding generation using OpenAI
- Vector storage in Pinecone

This process ensures high-quality semantic retrieval while keeping token usage efficient.

___

# 04. Query & Retrieval Pipeline <a name="rag-query"></a>

![Query Flow]("/posts/query-flow.jpg")

At query time:

- A chat trigger receives the user question
- The query is embedded
- Pinecone retrieves the most relevant chunks
- The AI agent generates a response using only retrieved context

If no relevant context is found, the agent safely declines to answer.

___

# 05. Application & Example Queries <a name="rag-application"></a>

The system is exposed through a **public n8n-hosted chat interface**:

![Chat UI]("/posts/chat-ui.jpg")

Example query:

> **What new camera features are introduced in iOS 18?**

The agent retrieves relevant sections from the documentation and produces a grounded, document-based response.

Unsupported questions are rejected with a clear fallback message.

___

# 06. Growth & Next Steps <a name="growth-next-steps"></a>

This project forms a strong foundation for:

- Documentation chatbots
- Product support assistants
- Internal knowledge bases
- No-code / low-code AI systems

Future work could extend this into a multi-document, multi-product RAG platform with richer UI and analytics.

___
