---
layout: post
title: Voice RAG Sales Assistant for E-commerce Product Support
image: "/img/posts/01-elevenlabs-agent.png"
tags: [Voice AI, RAG, n8n, Supabase, Twilio, ElevenLabs, OpenAI Embeddings]
---

In this project, I built a **voice-enabled Retrieval-Augmented Generation (RAG) sales assistant** for an e-commerce computer-accessories store. Customers can call a phone number, ask product-related questions in natural language, and receive concise spoken answers grounded in the store's current product catalogue.

The solution combines **ElevenLabs**, **Twilio**, **n8n**, **Supabase pgvector**, **OpenAI embeddings**, **OpenRouter**, and **Google Sheets**. It includes a separate data-ingestion workflow for keeping the vector knowledge base up to date and an agentic retrieval workflow for answering live customer questions without inventing unavailable product information.

---

## 🔗 Project Links

- **GitHub Repository:** [https://github.com/seyyednavid/voice-rag-sales-assistant](https://github.com/seyyednavid/voice-rag-sales-assistant)

---

# Table of Contents

- [00. Project Overview](#overview-main)
  - [Context](#overview-context)
  - [Actions](#overview-actions)
  - [Results](#overview-results)
  - [Growth / Next Steps](#overview-growth)
- [01. End-to-End Architecture](#system-architecture)
- [02. Voice Conversation Layer](#voice-layer)
- [03. Agentic RAG Workflow](#agentic-rag)
- [04. Product Data Ingestion](#data-ingestion)
- [05. Supabase Vector Knowledge Base](#vector-knowledgebase)
- [06. Retrieval Grounding and Safety](#retrieval-safety)
- [07. Customer Interaction Flow](#customer-flow)
- [08. Technical Decisions and Trade-Offs](#technical-decisions)
- [09. Current Limitations](#limitations)
- [10. Growth & Next Steps](#growth-next-steps)

___

# 00. Project Overview <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Customers often need quick help comparing products, checking prices, understanding features, or finding an accessory suitable for a particular use case. Traditional product pages and text-based chat interfaces can create friction, especially when a customer prefers to ask a question naturally by phone.

A voice assistant can make this interaction faster, but it must remain grounded in the current catalogue. A generic language model may otherwise invent product names, prices, stock information, compatibility details, or warranties that do not exist in the available data.

The goal of this project was to build a practical **voice-based product assistant** that combines natural conversation with retrieval from a controlled product knowledge base.

---

### Actions <a name="overview-actions"></a>

I designed and implemented a system that:

- Accepts inbound customer calls through a Twilio phone number
- Uses an ElevenLabs conversational voice agent for natural spoken interaction
- Sends product questions to an n8n webhook
- Uses an n8n AI Agent as the backend retrieval assistant
- Searches a Supabase Vector Store through a dedicated retrieval tool
- Generates OpenAI embeddings for product content
- Uses OpenRouter as the language-model provider for the backend agent
- Loads product data from Google Sheets through a separate ingestion workflow
- Stores product content and metadata in a PostgreSQL table with pgvector
- Clears old records before re-ingestion to prevent duplicate catalogue entries
- Supports follow-up questions such as “How much is it?”
- Returns a clear unavailable-product response when no matching item is found
- Keeps internal workflow and database details hidden from the customer

---

### Results <a name="overview-results"></a>

The completed prototype provides an end-to-end voice support workflow for catalogue-based product enquiries.

It can help customers:

- Find available computer accessories
- Ask for product recommendations
- Check product prices
- Compare suitable options
- Ask follow-up questions during the same conversation
- Receive concise answers designed for spoken delivery
- Avoid misleading answers when a product is not available in the catalogue

The project also separates **catalogue ingestion** from **live question answering**, making the system easier to maintain and update.

---

### Growth / Next Steps <a name="overview-growth"></a>

The next stage would add richer product metadata, structured price filtering, inventory availability, lead capture, CRM integration, scheduled catalogue synchronisation, and conversation analytics.

___

# 01. End-to-End Architecture <a name="system-architecture"></a>

The platform contains two connected pipelines:

1. A **customer-query pipeline** that handles phone conversations and product retrieval
2. A **data-ingestion pipeline** that converts catalogue records into searchable vector documents

### Customer Query Flow

```text
Customer Phone Call
→ Twilio Phone Number
→ ElevenLabs Voice Agent
→ n8n Webhook
→ n8n AI Agent
→ Supabase Vector Store
→ knowledgebase Table
→ n8n Response
→ ElevenLabs Voice Response
→ Customer
```

### Data Ingestion Flow

```text
Google Sheets Product Catalogue
→ n8n Data Ingest Workflow
→ Clear Existing Knowledge Base
→ Transform Product Records
→ Generate OpenAI Embeddings
→ Insert Documents into Supabase Vector Store
```

This separation ensures that catalogue preparation does not run during a live customer conversation.

___

# 02. Voice Conversation Layer <a name="voice-layer"></a>

The customer-facing layer is implemented with an ElevenLabs voice agent connected to a Twilio phone number.

![ElevenLabs voice sales agent configuration](/img/posts/01-elevenlabs-agent.png)

The voice agent is designed to be:

- Friendly and professional
- Concise enough for phone conversations
- Focused on product discovery and comparison
- Able to ask one short clarifying question when required
- Grounded in results returned by the product lookup tool

Its opening message introduces the assistant and immediately explains that it can help customers find accessories, check product details, and compare options.

The voice layer does not search the catalogue directly. Instead, it sends product questions to the n8n backend and speaks the returned customer-ready answer.

___

# 03. Agentic RAG Workflow <a name="agentic-rag"></a>

The live retrieval workflow is implemented in n8n.

![n8n Agentic RAG workflow](/img/posts/02-n8n-agentic-rag-workflow.jpg)

The workflow follows this sequence:

```text
Webhook
→ AI Agent
→ Supabase Vector Store Retrieval Tool
→ Respond to Webhook
```

The webhook receives a request in the following format:

```json
{
  "question": "Do you have a wireless mouse?"
}
```

The AI Agent uses an OpenRouter chat model and is instructed to call the product knowledge-base tool whenever the customer asks about:

- Product names
- Categories
- Prices
- SKUs
- Descriptions
- Specifications
- Recommendations
- Comparisons
- Product availability

The Supabase Vector Store is configured as an agent tool with a retrieval limit of five candidate documents. OpenAI embeddings are used to encode the customer query and compare it against the stored product vectors.

The final response is returned through the webhook in a short format suitable for text-to-speech.

___

# 04. Product Data Ingestion <a name="data-ingestion"></a>

A separate n8n workflow prepares the product catalogue for retrieval.

![n8n product data ingestion workflow](/img/posts/03-n8n-data-ingest-workflow.jpg)

The workflow runs through the following stages:

```text
Manual Trigger
→ Clear knowledgebase
→ Extract Products from Google Sheets
→ Transform Product Fields
→ Generate OpenAI Embeddings
→ Insert into Supabase Vector Store
```

The expected catalogue fields are:

| Field | Purpose |
|---|---|
| `Name` | Product name |
| `Category` | Product category |
| `SKU` | Product reference code |
| `Price` | Product price |
| `Description` | Searchable product description |

Each spreadsheet row is transformed into a searchable document such as:

```text
Product name: SwiftMouse Precision Pro
Category: Mouse
SKU: MS-101
Price: $49.99
Description: Wireless ergonomic mouse suitable for work and productivity.
```

The workflow also stores structured metadata for category, SKU, name, and price. This provides a foundation for future metadata filtering and structured product search.

Before loading the latest data, the workflow executes:

```sql
truncate table public.knowledgebase;
```

This prevents duplicate records when the ingestion pipeline is run repeatedly.

___

# 05. Supabase Vector Knowledge Base <a name="vector-knowledgebase"></a>

Supabase provides the PostgreSQL database and pgvector-based similarity search used by the RAG system.

![Supabase knowledgebase table](/img/posts/04-supabase-knowledgebase.png)

The main table is called `knowledgebase` and contains four core columns:

| Column | Responsibility |
|---|---|
| `id` | Unique identifier for each product document |
| `content` | Searchable product text |
| `metadata` | JSON metadata such as category, SKU, name, and price |
| `embedding` | Vector representation of the product content |

A database matching function enables similarity search against the stored vectors. The n8n retrieval tool uses this function indirectly through the Supabase Vector Store integration.

This design keeps the product catalogue independent from the language model and allows the underlying catalogue to be updated without changing the agent prompt.

___

# 06. Retrieval Grounding and Safety <a name="retrieval-safety"></a>

Grounded behaviour is a central design decision in this project.

The backend agent is instructed to:

- Always use the product knowledge-base tool for catalogue-related questions
- Answer only from retrieved product information
- Avoid inventing names, prices, stock status, warranties, delivery details, discounts, compatibility, or specifications
- Return a clear message when a matching product is unavailable
- Keep internal implementation details hidden from the customer
- Ask one short clarifying question when a follow-up reference is ambiguous

For example, when a customer asks about a product that is not present, the assistant responds with a message such as:

```text
That product is not available in the current catalogue.
```

This approach reduces hallucination risk and keeps the voice assistant aligned with the actual product data.

The repository is also sanitised and does not expose real API keys, webhook URLs, authentication tokens, phone numbers, database passwords, or customer data.

___

# 07. Customer Interaction Flow <a name="customer-flow"></a>

A typical conversation may follow this pattern:

```text
Customer: Do you have a wireless mouse?
Assistant: Yes. We have the SwiftMouse Precision Pro, a wireless ergonomic mouse designed for work and productivity. It is priced at $49.99.

Customer: How much is it?
Assistant: The SwiftMouse Precision Pro is $49.99.
```

The assistant can also support questions such as:

```text
Can you recommend a webcam for meetings?
```

```text
Which keyboard is suitable for office work?
```

```text
Can you compare these two options?
```

```text
Do you sell laptops?
```

When the customer asks a broad question, the voice agent can request one useful detail such as budget, preferred category, device type, or intended use before recommending an item.

___

# 08. Technical Decisions and Trade-Offs <a name="technical-decisions"></a>

### Voice agent separated from retrieval logic

The ElevenLabs agent manages conversation and speech, while n8n handles retrieval. This keeps the voice configuration simpler and makes the backend reusable across other channels.

### Independent ingestion workflow

Product ingestion is separated from customer-query execution. This reduces live-call latency and avoids repeatedly generating catalogue embeddings during conversations.

### Google Sheets as the catalogue source

Google Sheets makes product data easy to edit for a prototype or small store. However, it is not intended to replace a transactional inventory or commerce platform at larger scale.

### Full-table refresh before ingestion

Truncating the table provides simple duplicate prevention and ensures the vector store reflects the latest sheet. The trade-off is that every refresh rebuilds the complete knowledge base rather than applying incremental updates.

### Semantic retrieval instead of structured commerce search

Vector search supports natural-language product discovery and recommendation. However, exact numeric filters, sorting, real-time stock, and complex compatibility rules may require additional structured queries.

### Concise spoken responses

The system prioritises short answers that sound natural over the phone. More detailed information can still be provided when the customer explicitly asks for it.

___

# 09. Current Limitations <a name="limitations"></a>

- Product answers depend on the completeness and accuracy of the current Google Sheet
- Stock, delivery, warranty, discount, and compatibility details cannot be confirmed unless included in the catalogue
- The current knowledge-base refresh replaces all existing product documents
- The workflow is designed primarily for a demo or small product catalogue
- Advanced structured filtering and price sorting are not yet implemented
- Conversation memory depends on the configured voice-agent context and is not independently persisted in n8n
- The system does not yet create orders or capture sales leads
- Production monitoring, evaluation dashboards, and automated regression tests are not yet included

___

# 10. Growth & Next Steps <a name="growth-next-steps"></a>

The strongest future improvements would be:

- Add stock availability and inventory synchronisation
- Add brand, compatibility, colour, and use-case metadata
- Combine vector retrieval with structured SQL filtering
- Add numeric price filtering and sorting
- Add lead capture and CRM integration
- Create orders or shopping-cart requests from the conversation
- Schedule automatic product ingestion
- Replace full-table refreshes with incremental updates
- Store conversation history and customer preferences
- Add analytics for common questions and requested products
- Add automated test conversations and retrieval-quality evaluation
- Add latency, cost, and failed-call monitoring
- Support web chat and messaging channels through the same retrieval backend

This project demonstrates how **voice AI, agentic workflow automation, vector retrieval, database design, telephony, and grounded prompt engineering** can be combined into a practical e-commerce product-support system.

---

## Technology Stack

`ElevenLabs` · `Twilio` · `n8n` · `Supabase` · `PostgreSQL` · `pgvector` · `OpenAI Embeddings` · `OpenRouter` · `Google Sheets`

---

> This project is a portfolio-ready prototype designed for educational and demonstration purposes. Production use would require stronger authentication, monitoring, structured inventory integration, and broader automated testing.
