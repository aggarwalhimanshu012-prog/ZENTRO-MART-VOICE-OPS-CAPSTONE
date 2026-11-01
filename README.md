# ZENTRO-MART-VOICE-OPS-CAPSTONE
## 1. Overview
This repository implements ZentroMart's end-to-end grounded voice customer support agent across four progressive stages:
- **Step 1 (Prompt-Only Baseline):** `Qwen/Qwen2.5-3B-Instruct` prompted without company documents (19.2% strict accuracy).
- **Step 2 (Grounded RAG):** Section-level Markdown chunking (`kb/*.md`), multilingual semantic indexing (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`), product catalog resolution (`products.csv`), and business-day calendar calculation (`holidays.csv`), achieving **97.9% Retrieval Hit Rate @ 4** and **100% Hallucination Guard**.
- **Step 3 (Stateful Multi-Turn Engine):** Enforces caller identity verification (`IDV-2` / `IDV-3`) before revealing order details, executes `lookup_order` and `create_ticket` tools, and achieves **100.0% Tool Call Accuracy**, **99.0% Must-Include Accuracy**, **100.0% Must-Not-Include Guard**, and **100.0% Gold Section Retrieval Hit Rate** across all 96 turns (`PC002`–`PC024`).
- **Step 4 (Gnani Voice Wrapper):** Connects `GnaniSTTClient` and `GnaniTTSClient` (`model="timbre-v2.5"`, `voice="Kaveri"`) with spoken-number input normalization and ear-friendly TTS output normalization.

## 2. Installation & Setup

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
