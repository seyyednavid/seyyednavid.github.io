---
layout: post
title: Advanced RAG System with Evaluation and Optimisation
image: "/img/posts/advanced-rag-system-title-combined.png"
tags: [RAG, Generative AI, Hybrid Search, BM25, ChromaDB, LLM Evaluation]
---

In this project, I built and evaluated two **Retrieval-Augmented Generation (RAG)** pipelines: a conventional vector-search baseline and an advanced hybrid system designed to improve retrieval quality and answer accuracy.

The advanced pipeline combines **LLM-based semantic chunking, query rewriting, multi-query retrieval, ChromaDB vector search, BM25 keyword search, document enrichment, deduplication, and LLM reranking**. Both pipelines are measured with the same retrieval and answer-quality framework, making the improvements visible rather than relying only on subjective examples.

---

## 🔗 Project Links

- **GitHub Repository:** [https://github.com/seyyednavid/advanced-rag-system](https://github.com/seyyednavid/advanced-rag-system)

---

# Table of Contents

- [00. Project Overview](#overview-main)
  - [Context](#overview-context)
  - [Actions](#overview-actions)
  - [Results](#overview-results)
  - [Growth / Next Steps](#overview-growth)
- [01. Baseline RAG Pipeline](#baseline-rag)
- [02. Advanced RAG Architecture](#advanced-architecture)
- [03. Semantic Chunking and Enrichment](#semantic-chunking)
- [04. Query Rewriting and Multi-Query Retrieval](#query-rewriting)
- [05. Hybrid Retrieval and Reranking](#hybrid-retrieval)
- [06. Evaluation Framework](#evaluation-framework)
- [07. Results and Comparison](#results-comparison)
- [08. Embedding Visualisation](#embedding-visualisation)
- [09. Application Interface](#application-interface)
- [10. Trade-Offs and Limitations](#trade-offs)
- [11. Growth & Next Steps](#growth-next-steps)

___

# 00. Project Overview <a name="overview-main"></a>

### Context <a name="overview-context"></a>

A basic RAG system can retrieve semantically similar text, but vector similarity alone does not always return the best evidence. It may miss exact names, numbers, rare terms, or information split across several documents. Fixed-size chunking can also separate related ideas and weaken the context supplied to the language model.

The goal of this project was to investigate whether a more advanced retrieval pipeline could outperform a conventional vector-only baseline and to measure the difference using repeatable retrieval and answer-quality metrics.

---

### Actions <a name="overview-actions"></a>

I designed and implemented an evaluation-driven RAG workflow that:

- Establishes a baseline using fixed-size chunking and vector similarity search
- Creates semantically coherent chunks with an LLM
- Rewrites follow-up questions using conversation history
- Searches with both the original and rewritten query
- Combines ChromaDB vector retrieval with BM25 keyword retrieval
- Generates enriched summaries to support broader and multi-hop questions
- Merges and deduplicates results from multiple retrieval paths
- Uses an LLM reranker to select the strongest evidence
- Evaluates retrieval using MRR, nDCG, and coverage
- Evaluates generated answers for accuracy, completeness, and relevance
- Provides a local browser interface for testing questions and answers

---

### Results <a name="overview-results"></a>

The advanced pipeline outperformed the baseline across every measured metric.

| Metric | Baseline | Advanced | Improvement |
|---|---:|---:|---:|
| MRR | 0.7911 | **0.8757** | **+10.7%** |
| nDCG | 0.7918 | **0.8606** | **+8.7%** |
| Coverage | 92.8% | **93.8%** | **+1.0 percentage point** |
| Accuracy | 4.11 / 5 | **4.71 / 5** | **+14.6%** |
| Completeness | 3.99 / 5 | **4.21 / 5** | **+5.5%** |
| Relevance | 4.70 / 5 | **4.73 / 5** | **+0.6%** |

The strongest improvements were in **ranking quality** and **answer accuracy**, showing that better retrieval and context selection had a measurable effect on the final responses.

---

### Growth / Next Steps <a name="overview-growth"></a>

The next stage would focus on experiment tracking, latency and cost measurement, source citations, additional reranker comparisons, and containerised deployment.

___

# 01. Baseline RAG Pipeline <a name="baseline-rag"></a>

The baseline provides a controlled reference point for measuring the advanced system.

It uses:

- `RecursiveCharacterTextSplitter` with a chunk size of 500 and overlap of 200
- OpenAI embeddings
- ChromaDB vector storage
- Top-K semantic similarity search
- Retrieved chunks, conversation history, and the user question for answer generation

This approach is simple and effective for many use cases, but it relies entirely on dense vector similarity and divides documents according to character count rather than meaning.

![Baseline RAG evaluation results](/img/posts/Basic_RAG.jpg)

___

# 02. Advanced RAG Architecture <a name="advanced-architecture"></a>

The advanced system extends the baseline with specialised ingestion, retrieval, ranking, and evaluation stages.

The end-to-end flow is:

1. Source documents are processed with LLM-based semantic chunking
2. OpenAI embeddings are stored in a ChromaDB vector index
3. The same corpus is indexed for BM25 keyword retrieval
4. Structured summaries enrich the searchable knowledge base
5. Conversation context is used to rewrite ambiguous questions
6. Both the original and rewritten questions are searched
7. Vector, BM25, and enriched-summary results are combined
8. Duplicate evidence is removed
9. An LLM reranks the candidate chunks
10. The strongest Top-K context is passed to the answer model
11. Retrieval and answer quality are evaluated

This architecture addresses different retrieval failure modes instead of depending on a single search method.

___

# 03. Semantic Chunking and Enrichment <a name="semantic-chunking"></a>

Rather than splitting every document at a fixed character boundary, the advanced ingestion pipeline asks an LLM to identify semantically meaningful sections.

Each chunk contains:

- A headline
- A concise summary
- The original source text
- Source-aware metadata where available

This helps preserve coherent ideas and reduces the chance that important context is separated across unrelated chunks.

The pipeline also creates structured product summaries containing descriptions, features, clients, contracts, employees, and relationships between entities. These enriched records provide additional retrieval targets for broad questions and queries that require information from multiple sources.

___

# 04. Query Rewriting and Multi-Query Retrieval <a name="query-rewriting"></a>

Follow-up questions often contain pronouns or incomplete references, such as “What about its clients?” The query-rewriting stage uses conversation history to convert these questions into explicit, self-contained search queries.

The retriever then searches with:

- The user's original question
- The rewritten, context-aware question

Using both versions preserves the user's original intent while improving recall for ambiguous or conversational queries.

___

# 05. Hybrid Retrieval and Reranking <a name="hybrid-retrieval"></a>

The advanced retriever combines complementary search techniques.

### Vector Search

OpenAI embeddings and ChromaDB provide semantic matching, allowing the system to retrieve conceptually related passages even when the wording differs from the question.

### BM25 Keyword Search

BM25 strengthens retrieval for exact terms such as names, identifiers, numerical values, salaries, and rare keywords that dense embeddings may underweight.

### Deduplication and LLM Reranking

Candidate chunks from the original query, rewritten query, vector index, BM25 index, and enriched summaries are merged and deduplicated. An LLM reranker then prioritises evidence based on factual relevance, completeness, multi-part question coverage, and overall evidence quality.

Only the strongest context is passed to the generation stage, reducing duplicated or weak evidence in the final prompt.

___

# 06. Evaluation Framework <a name="evaluation-framework"></a>

The project evaluates both the retrieval stage and the final generated answer.

### Retrieval Metrics

- **MRR (Mean Reciprocal Rank):** measures how highly the first relevant result appears
- **nDCG (Normalised Discounted Cumulative Gain):** measures the ranking quality of multiple relevant results
- **Coverage:** measures how often the retriever returns the required evidence

### Answer-Quality Metrics

- **Accuracy:** factual correctness of the response
- **Completeness:** coverage of the information requested by the user
- **Relevance:** alignment between the response and the question

The baseline and advanced pipelines use the same evaluation process, allowing a direct comparison between the two designs.

___

# 07. Results and Comparison <a name="results-comparison"></a>

![Advanced RAG evaluation results](/img/posts/Advanced.jpg)

The advanced system achieved:

- **MRR:** 0.8757
- **nDCG:** 0.8606
- **Coverage:** 93.8%
- **Accuracy:** 4.71 / 5
- **Completeness:** 4.21 / 5
- **Relevance:** 4.73 / 5

The results indicate that the advanced pipeline did more than retrieve additional chunks. It ranked useful evidence more effectively and provided the generation model with higher-quality context, producing a substantial improvement in answer accuracy.

___

# 08. Embedding Visualisation <a name="embedding-visualisation"></a>

The project also compares the structure of embedding spaces generated by Hugging Face and OpenAI models.

### Hugging Face Embeddings

![Hugging Face embeddings in 2D](/img/posts/HuggingFaceEmbeddings_all-MiniLM-L6-v2.png)

![Hugging Face embeddings in 3D](/img/posts/HuggingFaceEmbeddings_all-MiniLM-L6-v2-3D.png)

### OpenAI Embeddings

![OpenAI embeddings in 2D](/img/posts/text-embedding-3-large.png)

![OpenAI embeddings in 3D](/img/posts/text-embedding-3-large-3d.png)

For this dataset, the OpenAI embedding plots show clearer clustering and stronger topic separation. This observation is consistent with the retrieval results, although visual separation alone is not treated as a substitute for quantitative evaluation.

___

# 09. Application Interface <a name="application-interface"></a>

The project includes a local browser application for testing the complete pipeline.

![Advanced RAG application interface](/img/posts/app.jpg)

Users can submit natural-language questions and inspect the generated answers. The interface provides a practical way to test retrieval behaviour and compare the system's output with the evaluation findings.

___

# 10. Trade-Offs and Limitations <a name="trade-offs"></a>

The advanced architecture improves retrieval quality, but every additional stage introduces a cost.

| Technique | Benefit | Trade-off |
|---|---|---|
| Semantic chunking | More coherent context | Higher ingestion cost |
| Query rewriting | Better conversational recall | Additional LLM call |
| Multi-query retrieval | Broader evidence coverage | More retrieval work |
| BM25 | Stronger exact-term matching | Additional index and compute |
| LLM enrichment | Better broad and multi-hop retrieval | Extra preprocessing |
| LLM reranking | Higher-quality final context | Increased latency and cost |

Current limitations include:

- The application runs locally and is not yet a production deployment
- The dataset is domain-specific
- Evaluation scores may vary across models and embedding providers
- LLM-based answer evaluation can introduce judge-model bias
- Authentication, production monitoring, and tracing are not yet included
- The current evaluation does not report retrieval latency or per-query cost

___

# 11. Growth & Next Steps <a name="growth-next-steps"></a>

Future improvements include:

- Automated experiment tracking across pipeline configurations
- Query-level error analysis for failed retrieval cases
- Retrieval latency, token usage, and cost benchmarks
- Source citations and source highlighting in the interface
- Comparison with RAGAS or another evaluation framework
- Langfuse or OpenTelemetry tracing
- Configurable retrieval profiles
- Vector-database and reranker-model comparisons
- Docker-based local deployment

This project demonstrates an **evaluation-driven RAG development process**: establish a baseline, introduce targeted retrieval improvements, measure their effect, and make the trade-offs visible.

___
