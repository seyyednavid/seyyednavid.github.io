---
layout: post
title: Building a Semantic Course Search Engine Using Weighted BERT Embeddings
image: "/posts/semantic-course-search-title-img.png"
tags: [NLP, Semantic Search, Vector Databases, BERT, Pinecone, Streamlit]
---

In this project, we build a **production-style semantic search system** for online course content.  
The system allows users to search for relevant course sections using **natural language queries**, retrieving results based on **semantic meaning rather than keywords**, powered by **weighted BERT embeddings** and a vector database.

We explore multiple embedding strategies and data granularities, evaluate their effectiveness, and deploy the best-performing approach as an **interactive web application**.

---

## 🔗 Project Links

- **GitHub Repository:**  [https://github.com/seyyednavid/Semantic-course-search](https://github.com/seyyednavid/Semantic-course-search)
- **Live Application / Demo:**  [https://semantic-course-search6264.streamlit.app/](https://semantic-course-search6264.streamlit.app/)

---

# Table of Contents

- [00. Project Overview](#overview-main)
  - [Context](#overview-context)
  - [Actions](#overview-actions)
  - [Results](#overview-results)
  - [Growth / Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
- [02. Semantic Search Overview](#semantic-overview)
- [03. Embedding Strategies](#embedding-strategies)
- [04. Vector Database Design](#vector-database)
- [05. Weighted Semantic Querying](#weighted-querying)
- [06. Application & Examples](#application)
- [07. Growth & Next Steps](#growth-next-steps)

___

# 00. Project Overview <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Online learning platforms contain large volumes of educational content, often organised into courses and fine-grained sections.  
Traditional keyword-based search systems struggle to surface relevant content when users phrase questions differently from the original text.

Examples of such queries include:

- *“AI applications for business success”*  
- *“Regression in Python”*  
- *“Data science course”*

Answering these accurately requires understanding **semantic intent**, not just matching words.

---

### Actions <a name="overview-actions"></a>

We built a semantic search pipeline that:

- Embeds course content using modern sentence-transformer models  
- Stores embeddings in a vector database (Pinecone)  
- Retrieves the most semantically similar content using cosine similarity  
- Explores different **data granularities** (course-level vs section-level)  
- Evaluates lightweight and transformer-based embedding models  
- Deploys the best-performing approach as a Streamlit application  

---

### Results <a name="overview-results"></a>

The final system:

- Consistently retrieves highly relevant course sections  
- Outperforms keyword search for conceptual queries  
- Demonstrates clear improvements when moving from course-level to section-level indexing  
- Achieves the best results using **weighted BERT embeddings**

The deployed application allows users to interactively explore these results in real time.

---

### Growth / Next Steps <a name="overview-growth"></a>

Potential future improvements include:

- Quantitative evaluation metrics (Precision@K, Recall@K)  
- Query latency benchmarking across models  
- Model selection within the UI  
- User feedback loops to refine relevance  

___

# 01. Data Overview <a name="data-overview"></a>

Two datasets were used in this project:

1. **Course-level descriptions**  
   One row per course, containing high-level summaries.

2. **Section-level descriptions**  
   One row per course section, containing fine-grained instructional content.

The section-level dataset enables more precise retrieval, as semantic similarity is computed over smaller, more focused units of text.

___

# 02. Semantic Search Overview <a name="semantic-overview"></a>

Semantic search differs from keyword search by representing text as **dense vectors** that capture meaning.

With vector-based search:

1. Text is embedded into a numeric vector space  
2. Similar meanings map to nearby vectors  
3. Queries retrieve results using similarity metrics such as cosine similarity  

This allows the system to retrieve relevant content even when the query wording differs from the source text.

___

# 03. Embedding Strategies <a name="embedding-strategies"></a>

We evaluated four embedding strategies:

1. **Course-level MiniLM embeddings**  
2. **Section-level MiniLM embeddings**  
3. **Section-level BERT embeddings (unweighted)**  
4. **Section-level BERT embeddings (weighted)**  

Moving from course-level to section-level indexing significantly improved precision.  
Using a stronger transformer model further improved semantic understanding.

___

# 04. Vector Database Design <a name="vector-database"></a>

All embeddings are stored in **Pinecone**, a managed vector database optimised for similarity search.

Key design choices:

- Cosine similarity as the distance metric  
- Section-level documents as the primary retrieval unit  
- Metadata storage for course name, section name, and descriptions  

This design allows fast and scalable semantic retrieval.

___

# 05. Weighted Semantic Querying <a name="weighted-querying"></a>

This weighted embedding strategy represents the key improvement over standard semantic search and is the method used in the deployed application.

The final approach introduces **weighted semantic query embeddings**.

Instead of embedding the user query once, we:

1. Encode the raw user query  
2. Encode a contextualised version of the query  
3. Combine both embeddings using weighted averaging  

This reinforces the core semantic intent while preserving contextual relevance, improving retrieval quality for short or ambiguous queries.

This approach proved especially effective for short, ambiguous, or high-level user queries.


___

# 06. Application & Examples <a name="application"></a>

The final system is deployed as a **Streamlit web application**.

Users can:

- Enter natural-language queries  
- Retrieve the most relevant course sections  
- Inspect similarity scores  
- Expand detailed section descriptions  

Example queries include:

- *“technical analysis indicators”*  
- *“support and resistance levels”*  
- *“momentum oscillators explained”*  

The system consistently retrieves conceptually relevant sections, even when keyword overlap is minimal.

___

#  07. Growth & Next Steps <a name="growth-next-steps"></a>

Future enhancements may include:

- Adding user feedback to re-rank results  
- Hybrid search combining keyword and vector similarity  
- Incremental re-indexing pipelines  
- Advanced UI filtering and analytics  

This project demonstrates how **semantic search and vector databases** can be combined to build practical, user-facing information retrieval systems.

___
