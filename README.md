# 📰 Universal AI Investigative Journalist & Knowledge Graph Engine

A state-of-the-art deep research agentic workflow that scrapes the web, filters noise, extracts disambiguated facts, verifies them through a strict Critic Loop to eliminate hallucinations, builds interactive knowledge graphs, and compiles a comprehensive investigative report illustrated with locally-generated FLUX.1 images.

---

## 🚀 Key Features

* **Advanced Web Discovery & Stealth Crawling**: Leverages DuckDuckGo Search (DDGS) to discover top references and `Crawl4AI` (with stealth mode, anti-bot mechanisms, and dynamic pruning filters) to extract clean markdown corpora.
* **4-Stage Zero-Hallucination Pipeline**:
  1. **Stage 1: Noise & Ad Filter**: Separates high-value content from layout junk, navigation headers, and off-topic sections.
  2. **Stage 2: Fact Disambiguation**: Converts raw unstructured text into atomic, self-contained factual sentences, replacing ambiguous pronouns with proper nouns.
  3. **Stage 3: Candidate & Critic Verification Loop**: 
     * **Extraction (3A)**: Structures facts into Subject-Predicate-Object triples.
     * **Fact-Critic (3B)**: A rigorous critic agent cross-examines every single triple against the source corpus. Triples without direct textual evidence are instantly purged.
  4. **Stage 4: Grounded Synthesis Engine**:
     * **Outline Planner (4A)**: Editorial planning dividing triples into distinct thematic chapters.
     * **Synthesis (4B)**: Drafts full chapters grounded strictly in the verified facts. Every claim is cite-backed with its corresponding triple ID (e.g. `[T1]`, `[T2]`).
* **Interactive Knowledge Graph Networks**: Renders dynamic, responsive, and color-coded network graphs inside the UI using `pyvis`, saved as standalone HTML components.
* **Local FLUX.1 Image Generation**: Automatically generates context-aware illustrations for each chapter using a resource-optimized 4-bit quantized pipeline (`FLUX.1-schnell` with BitsAndBytes NF4 quantization) running locally.

---

## 🛠️ Setup & Installation

### 1. Install Dependencies
Make sure you have Python (version 3.10 or 3.11 is recommended) and the required packages installed. Run:

```bash
pip install -r requirement.txt
```

Additionally, complete the browser setup for the `Crawl4AI` engine by running:
```bash
playwright install
```

---

### 2. Configure the LLM (Ollama with Qwen)
The application relies on a local Ollama instance for the multi-stage research and verification pipeline.

1. Download and install [Ollama](https://ollama.com/).
2. Pull the **Qwen 2.5 (7B)** model (which is configured in the code as `qwen2.5:7b`):
   ```bash
   ollama pull qwen2.5:7b
   ```
3. Keep the Ollama server running locally at its default address (`http://localhost:11434`).

---

### 3. Pre-Download & Test the FLUX.1 Image Model
To avoid downloading large model weights while running the main Streamlit application, pre-cache the 4-bit quantized FLUX.1 pipeline using the provided test utility. 

Run the test script:
```bash
python download_and_test_image_model.py
```

* **What this does:**
  * Downloads the base model config/text-encoder from `black-forest-labs/FLUX.1-schnell`.
  * Downloads the 4-bit quantized transformer from `Keffisor21/flux1-schnell-bnb-nf4`.
  * Assembles the pipeline using sequential CPU offloading and VAE slicing/tiling to fit comfortably within consumer RAM/VRAM.
  * Generates a sample image named `test_local_flux_4bit.png` to verify successful setup.

---

## 🏃 How to Run the App

Launch the Streamlit dashboard by executing:

```bash
streamlit run app.py
```

### Configuration Options in the UI:
* **Maximum Web Links to Explore**: Choose how many search sources to retrieve.
* **Extraction Window Size**: Word limits for text segmentation.
* **Model Temperature**: Control creativity vs. deterministic factualness.
* **Number of Chapters**: Divide the final report into up to 6 distinct chapters.
* **Generate Local FLUX.1 Images**: Toggle on/off image illustration for each chapter.
