---
layout: post
title: AI Meeting Assistant
image: "/img/posts/ai-meeting-assistant.jpg"
tags: [AI Engineering, Speech-to-Text, LLM, OpenAI, Whisper, Ollama, Gradio]
---

In this project, I built an **AI Meeting Assistant** that converts meeting audio into structured meeting minutes. The application combines speech-to-text transcription, large-language-model summarisation, structured Markdown generation, model selection, fallback handling, and a Gradio interface into one end-to-end workflow.

The system supports both **cloud-based and local model options**: OpenAI or Hugging Face Whisper for transcription, and OpenAI GPT or a local Qwen 2.5 model through Ollama for meeting-minute generation.

---

## 🔗 Project Links

- **GitHub Repository:** [https://github.com/seyyednavid/ai-meeting-assistant](https://github.com/seyyednavid/ai-meeting-assistant)

---

# Table of Contents

- [00. Project Overview](#overview-main)
  - [Context](#overview-context)
  - [Actions](#overview-actions)
  - [Results](#overview-results)
  - [Growth / Next Steps](#overview-growth)
- [01. System Architecture](#system-architecture)
- [02. Audio Transcription](#audio-transcription)
- [03. Structured Meeting Minutes](#meeting-minutes)
- [04. Local and Cloud Model Options](#model-options)
- [05. Pipeline Orchestration and Fallback](#pipeline-orchestration)
- [06. Application Interface](#application-interface)
- [07. Output and Processing Details](#output-processing)
- [08. Reliability and File Handling](#reliability-file-handling)
- [09. Technical Decisions and Trade-Offs](#technical-decisions)
- [10. Current Limitations](#limitations)
- [11. Growth & Next Steps](#growth-next-steps)

___

# 00. Project Overview <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Meeting recordings often contain useful decisions, discussion points, and follow-up tasks, but reviewing long audio files manually is time-consuming. Converting a recording into usable documentation typically requires several separate steps: transcription, summarisation, action-item extraction, formatting, and file export.

The goal of this project was to build a practical AI application that combines these stages into one modular workflow while also demonstrating the trade-offs between cloud APIs and locally hosted open-source models.

---

### Actions <a name="overview-actions"></a>

I designed and implemented a workflow that:

- Accepts uploaded meeting audio through a Gradio interface
- Supports OpenAI transcription using `gpt-4o-mini-transcribe`
- Supports local speech recognition using Hugging Face `openai/whisper-base`
- Converts the transcript into structured meeting minutes
- Supports OpenAI `gpt-4o-mini` for cloud-based summarisation
- Supports local `qwen2.5:3b` through Ollama
- Falls back to OpenAI summarisation if the local Ollama model fails
- Extracts meeting overview, summary, discussion points, decisions, action items, and takeaways
- Avoids inventing missing dates, attendees, owners, or deadlines
- Displays the complete transcript and generated meeting minutes
- Reports model choices, processing time, transcript length, and output filename
- Exports the generated meeting minutes as a Markdown file
- Uses temporary audio files and removes them after processing

---

### Results <a name="overview-results"></a>

The completed application provides an end-to-end meeting-processing workflow with:

- Two transcription options: OpenAI and local Whisper
- Two summarisation options: OpenAI GPT and local Qwen
- Structured professional meeting minutes
- Clear separation between decisions and action items
- Explicit handling of missing information
- Automatic Markdown export
- Local-model fallback handling
- Processing metadata for easier testing and comparison
- A browser-based Gradio interface for non-technical users

The result is a modular AI engineering project that demonstrates how speech recognition, LLM summarisation, local models, cloud APIs, file handling, and user-interface design can be combined into a single application.

---

### Growth / Next Steps <a name="overview-growth"></a>

Potential future improvements include:

- Speaker diarisation
- Long-transcript chunking and hierarchical summarisation
- Docker support
- Cloud deployment
- Additional transcription and summarisation models
- Meeting-level search and history
- Evaluation of transcription and summarisation quality
- Export to additional formats such as PDF or DOCX

___

# 01. System Architecture <a name="system-architecture"></a>

The application separates the workflow into transcription, summarisation, orchestration, and user-interface layers.

At a high level:

```text
Meeting Audio
    ↓
Transcription
    ├── OpenAI GPT-4o Mini Transcribe
    └── Hugging Face Whisper Base
    ↓
Transcript
    ↓
Meeting Minutes Generation
    ├── OpenAI GPT-4o Mini
    └── Local Qwen 2.5 3B via Ollama
    ↓
Results
    ├── Full Transcript
    ├── Structured Meeting Minutes
    ├── Processing Details
    └── Markdown Download
```

The Python modules are separated by responsibility:

```text
ai-meeting-assistant/
│
├── app/
│   ├── transcription.py
│   ├── summarizer.py
│   └── pipeline.py
│
├── ui/
│   └── gradio_app.py
│
├── outputs/
├── temp_audio/
├── requirements.txt
└── README.md
```

This modular structure keeps model-specific logic separate from orchestration and interface code.

___

# 02. Audio Transcription <a name="audio-transcription"></a>

The transcription layer supports two different execution strategies.

### OpenAI Transcription

The cloud option uses:

```text
gpt-4o-mini-transcribe
```

The application uploads the audio file to the OpenAI transcription API and returns the resulting transcript.

### Hugging Face Whisper

The local option uses:

```text
openai/whisper-base
```

The Whisper pipeline is loaded lazily, meaning the model is initialised only when it is first required and can then be reused for later requests.

The local transcription configuration includes:

```text
chunk_length_s = 30
stride_length_s = 5
return_timestamps = True
```

This provides an open-source alternative that can run without using a transcription API, although CPU execution can be significantly slower for longer recordings.

___

# 03. Structured Meeting Minutes <a name="meeting-minutes"></a>

The summarisation layer converts the raw transcript into a consistent meeting-document format.

The generated Markdown contains:

```text
# Meeting Minutes

## Meeting Overview
- Date
- Location
- Attendees

## Summary

## Key Discussion Points

## Decisions

## Action Items

## Takeaways
```

The prompt includes explicit safeguards designed to reduce unsupported information:

- Missing dates, locations, attendees, owners, or deadlines are marked as not specified
- Decisions are treated separately from action items
- A decision is not converted into an action item unless a follow-up task is clearly stated
- Owners and deadlines are not invented
- If no follow-up tasks are found, the output explicitly reports that no clear action items were identified

___

# 04. Local and Cloud Model Options <a name="model-options"></a>

A core design goal was to make the pipeline configurable across both hosted and local models.

| Stage | Cloud option | Local option |
|---|---|---|
| Transcription | OpenAI `gpt-4o-mini-transcribe` | Hugging Face `openai/whisper-base` |
| Summarisation | OpenAI `gpt-4o-mini` | `qwen2.5:3b` through Ollama |

This creates a practical comparison between managed API services and local open-source inference within the same application.

___

# 05. Pipeline Orchestration and Fallback <a name="pipeline-orchestration"></a>

The `pipeline.py` module coordinates the complete workflow:

```text
Audio File
   ↓
Selected Transcription Model
   ↓
Transcript
   ↓
Selected Summarisation Model
   ↓
Meeting Minutes
```

The summarisation stage includes automatic fallback behaviour. If the local Ollama/Qwen request fails, the application logs the failure and switches to OpenAI summarisation.

```text
Local Qwen
   ↓
Success? ── Yes → Return meeting minutes
   │
   No
   ↓
Fallback to OpenAI
```

___

# 06. Application Interface <a name="application-interface"></a>

The project includes a Gradio browser interface that brings the complete workflow together.

![AI Meeting Assistant application interface](/img/posts/app-ui.jpg)

The user can:

- Upload a meeting recording
- Choose the transcription model
- Choose the summarisation model
- Generate meeting minutes
- View the full transcript
- View the structured meeting minutes
- Inspect processing information
- Download the result as a Markdown file

___

# 07. Output and Processing Details <a name="output-processing"></a>

For each successful request, the application reports:

- Transcription model
- Summarisation model
- Processing time
- Transcript length
- Output filename

Generated meeting minutes are saved using timestamped filenames such as:

```text
meeting_minutes_YYYYMMDD_HHMMSS.md
```

___

# 08. Reliability and File Handling <a name="reliability-file-handling"></a>

The application includes several safeguards around file processing and model execution:

- Uploaded audio is copied to a temporary directory using a UUID-based filename
- Temporary audio is removed after processing, including when an exception occurs
- Transcript text is HTML-escaped before display
- File copying is retried when temporary permission errors occur
- Errors are returned with model choices and processing time for easier debugging

___

# 09. Technical Decisions and Trade-Offs <a name="technical-decisions"></a>

### Separate transcription and summarisation modules

Speech recognition and summarisation are isolated into different modules, making it easier to replace or extend models later.

### Local and hosted model support

Supporting both execution styles demonstrates the trade-offs between speed, API cost, privacy, and local compute requirements.

### Lazy loading for Whisper

The Whisper model is loaded only when required and then reused.

### Conservative meeting-minute prompt

The prompt explicitly distinguishes decisions from action items and avoids inventing missing information.

### Transcript truncation

The current implementation limits transcript size before summarisation:

- OpenAI summarisation: approximately 20,000 characters
- Local Qwen summarisation: approximately 12,000 characters

This helps control latency and context-size issues but can omit content from very long meetings.

___

# 10. Current Limitations <a name="limitations"></a>

- Hugging Face Whisper can be slow on CPU for long recordings
- Local Qwen inference can be slower than cloud-based OpenAI summarisation
- Very long transcripts are truncated instead of processed in chunks
- Speaker diarisation is not yet included
- There is no persistent meeting-history database
- The application is currently designed for local execution rather than production deployment
- Formal transcription and summarisation quality benchmarks are not yet included

___

# 11. Growth & Next Steps <a name="growth-next-steps"></a>

Future improvements include:

- Speaker diarisation
- Transcript chunking for longer meetings
- Hierarchical summarisation
- Configurable prompt templates for different meeting types
- Meeting history, search, and tagging
- Automated evaluation of transcription and meeting-minute quality
- Docker support
- Cloud deployment
- PDF and DOCX export
- Additional local models for side-by-side comparison
- Authentication and multi-user support

This project demonstrates a practical **AI Engineering workflow** that combines speech-to-text, LLM summarisation, configurable local and cloud models, fallback behaviour, safe file handling, and an interactive user interface.

---

## Technology Stack

`Python` · `Gradio` · `OpenAI API` · `GPT-4o Mini` · `GPT-4o Mini Transcribe` · `Hugging Face Transformers` · `Whisper` · `Qwen 2.5` · `Ollama` · `PyTorch` · `FFmpeg` · `Markdown`

---
