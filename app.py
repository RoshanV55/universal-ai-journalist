import os
import json
import ast
import re
import asyncio
import requests
import torch
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import T5EncoderModel, BitsAndBytesConfig
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from ddgs import DDGS
from pyvis.network import Network

def safe_parse_json(json_str: str):
    json_str = json_str.strip()
    try:
        return json.loads(json_str)
    except Exception:
        pass
    try:
        processed = re.sub(r'\btrue\b', 'True', json_str)
        processed = re.sub(r'\bfalse\b', 'False', processed)
        processed = re.sub(r'\bnull\b', 'None', processed)
        return ast.literal_eval(processed)
    except Exception:
        pass
    try:
        cleaned = re.sub(r',\s*([\]}])', r'\1', json_str)
        return json.loads(cleaned)
    except Exception:
        pass
    return json.loads(json_str)


# --- Environment Configuration ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- Model Configurations ---
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# --- Streamlit Page Setup ---
st.set_page_config(page_title="AI Investigative Journalist & KG Generator", page_icon="📰", layout="wide")

st.title("📰 Universal AI Investigative Journalist & Knowledge Graph Engine")
st.caption("Deep research via Crawl4AI, Fact Disambiguation, Critic-Verified Knowledge Graphs, and local FLUX.1 generation.")

# --- Local FLUX.1 Model Loader (Cached) ---
@st.cache_resource
def load_local_flux_pipeline():
    st.info("⚡ Loading local 4-Bit FLUX.1 Pipeline into memory... (This runs once)")
    
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )

    text_encoder_8bit = T5EncoderModel.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        subfolder="text_encoder_2",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )

    transformer_4bit = FluxTransformer2DModel.from_pretrained(
        "Keffisor21/flux1-schnell-bnb-nf4",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu"
    )

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        text_encoder_2=text_encoder_8bit,
        transformer=transformer_4bit,
        torch_dtype=torch.bfloat16
    )

    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    return pipe

# --- UI Sidebar ---
st.sidebar.header("Research & Generation Configurations")
max_links = st.sidebar.slider("Maximum Web Links to Explore", min_value=2, max_value=12, value=5)
chunk_size = st.sidebar.slider("Extraction Window Size (Words)", min_value=800, max_value=2500, value=1500)
temperature = st.sidebar.slider("Model Temperature (Creativity Control)", min_value=0.0, max_value=1.0, value=0.1)
num_chapters = st.sidebar.slider("Number of Chapters to Generate", min_value=2, max_value=6, value=3)
enable_images = st.sidebar.checkbox("Generate Local FLUX.1 Images for Chapters", value=True)

# Main Topic Input
topic = st.text_area(
    "Enter Deep Research Target / Investigative Topic:",
    value="Quantum Computing advances and challenges",
    height=70
)

# --- Helper Functions ---
async def search_web(query: str, max_links: int) -> list[str]:
    links = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query=query, max_results=max_links))
            for r in results:
                url = r.get("href")
                if url:
                    links.append(url)
    except Exception as e:
        st.sidebar.error(f"Search Extraction Engine Alert: {e}")
    return links

def chunk_text(text: str, max_words: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

async def call_ollama(prompt: str, temp: float = 0.0, timeout: int = 180) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "options": {"temperature": temp},
                "stream": False
            },
            timeout=timeout
        )
        if response.status_code == 200:
            return response.json().get('response', '')
        return f"[Error: Ollama returned status {response.status_code}]"
    except Exception as e:
        return f"[Error calling Ollama: {e}]"

def generate_local_flux_image(pipe: FluxPipeline, prompt: str) -> Image.Image | None:
    try:
        image = pipe(
            prompt,
            height=512,
            width=512,
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256
        ).images[0]
        return image
    except Exception as e:
        st.error(f"Local FLUX.1 Generation Error: {e}")
        return None

# --- Pipeline Modules: 4-Stage Zero-Hallucination Architecture ---

# Stage 1: Noise & Ad Filter
async def filter_relevant_chunks(chunks: list[str], topic: str) -> list[str]:
    """Stage 1: Filters out scrapings from sidebars, ads, and off-topic noise before processing."""
    relevant_chunks = []
    for chunk in chunks:
        filter_prompt = f"""
        Task: Determine if the following text contains factual information directly relevant to the topic: "{topic}".
        If it consists mostly of navigation menus, advertisements, unrelated news tickers, or irrelevant side-content, reply NO.
        Otherwise, reply YES.

        TEXT SEGMENT:
        {chunk[:800]}

        Reply ONLY with 'YES' or 'NO'.
        """
        response = await call_ollama(filter_prompt, temp=0.0)
        if "YES" in response.upper():
            relevant_chunks.append(chunk)
            
    return relevant_chunks if relevant_chunks else chunks

# Stage 2: Fact Disambiguation
async def disambiguate_facts(raw_text: str) -> list[str]:
    """Stage 2: Fact Disambiguation - Converts unstructured text into clear, unambiguous statements."""
    prompt = f"""
    Task: Convert the following raw text into a list of standalone, unambiguous facts.
    
    RULES:
    1. Replace all ambiguous pronouns (he, she, it, they, the company, the party) with explicit proper nouns or terms.
    2. Split complex or compound sentences into single atomic claims.
    3. Remove speculative opinions, clickbait, or unverified claims.
    4. Each line must be a complete self-contained factual statement.

    RAW TEXT:
    {raw_text}
    """
    response = await call_ollama(prompt, temp=0.0)
    facts = [line.strip("- ").strip() for line in response.split("\n") if line.strip()]
    return facts

# Stage 3A: Universal Knowledge Graph Extraction
async def build_knowledge_graph(disambiguated_facts: list[str]) -> dict:
    """Stage 3A: Universal Knowledge Graph Engine - Extracts Candidate JSON Triples for any topic."""
    facts_block = "\n".join(f"- {f}" for f in disambiguated_facts)
    
    prompt = f"""
    You are an expert Knowledge Graph Extraction Engine.
    Extract Subject-Predicate-Object triples from the provided factual statements.

    OUTPUT FORMAT: Return strictly valid JSON in this exact structure:
    {{
      "nodes": [
        {{"id": "Entity Name", "type": "Person/Organization/Location/Concept/Event/Metric/Date/Technology"}}
      ],
      "edges": [
        {{"source": "Entity Name", "target": "Target Entity", "relation": "relationship_predicate"}}
      ]
    }}

    CRITICAL RULES:
    1. Output strictly valid JSON. Do not include markdown commentary around the JSON.
    2. Ensure every source and target entity in "edges" exists in "nodes".

    FACTS TO PROCESS:
    {facts_block}
    """
    
    response = await call_ollama(prompt, temp=0.0)
    
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start != -1 and json_end != -1:
            graph_data = safe_parse_json(response[json_start:json_end])
            return graph_data
        else:
            st.warning("No valid JSON structure detected in Ollama response.")
            return {"nodes": [], "edges": []}
    except Exception as e:
        st.warning(f"Error parsing graph JSON: {e}")
        st.error(f"Raw Ollama Response: {repr(response)}")
        return {"nodes": [], "edges": []}

# Stage 3B: Critic Loop Verification Agent
async def verify_triples_critic(raw_text_corpus: str, unverified_graph: dict) -> dict:
    """Stage 3B: Critic Loop Agent - Cross-checks triples against raw text."""
    edges = unverified_graph.get("edges", [])
    nodes = unverified_graph.get("nodes", [])
    
    if not edges:
        return {"nodes": nodes, "edges": []}

    triples_formatted = "\n".join(
        [f"ID {idx}: ({e.get('source')}) ──[{e.get('relation')}]──► ({e.get('target')})"
         for idx, e in enumerate(edges)]
    )

    critic_prompt = f"""
    You are a Strict Fact-Checking Verification Agent.
    Your job is to cross-examine extracted Knowledge Graph Triples against the RAW SOURCE TEXT.

    CRITICAL RULES:
    1. If a triple's relationship, numbers, dates, or names are 100% EXPLICITLY supported by the raw text, mark it as 1.
    2. If a triple contains invented numbers, incorrect entity links, ads, or unsupported claims, mark it as 0.
    3. Output STRICTLY a JSON array of objects with "id" and "status" (1 for PASS, 0 for FAIL).

    RAW SOURCE TEXT:
    {raw_text_corpus[:4000]}

    TRIPLES TO VERIFY:
    {triples_formatted}

    OUTPUT FORMAT (Return JSON ONLY):
    [
      {{"id": 0, "status": 1}},
      {{"id": 1, "status": 0}}
    ]
    """

    response = await call_ollama(critic_prompt, temp=0.0)

    try:
        json_start = response.find("[")
        json_end = response.rfind("]") + 1
        if json_start != -1 and json_end != -1:
            verdict_list = safe_parse_json(response[json_start:json_end])
            
            passed_ids = {item["id"] for item in verdict_list if item.get("status") == 1}
            verified_edges = [edge for idx, edge in enumerate(edges) if idx in passed_ids]
            
            active_node_names = set()
            for edge in verified_edges:
                active_node_names.add(edge.get("source"))
                active_node_names.add(edge.get("target"))

            verified_nodes = [n for n in nodes if n.get("id") in active_node_names]

            return {"nodes": verified_nodes, "edges": verified_edges}
            
    except Exception as e:
        st.warning(f"Verification parser error: {e}. Falling back to default graph.")
        return unverified_graph

    return unverified_graph

# Stage 4A: Universal Outline Planner Agent
async def create_chapter_outline(topic: str, num_chapters: int, graph_data: dict) -> list[str]:
    """Dynamically divides knowledge triples across any domain into distinct thematic chapters."""
    edges = graph_data.get("edges", [])
    if not edges:
        return [f"General Overview of {topic}" for _ in range(num_chapters)]

    kg_triples = "\n".join(
        [f"[T{idx}] ({e.get('source')}) ──[{e.get('relation')}]──► ({e.get('target')})"
         for idx, e in enumerate(edges)]
    )
    
    prompt = f"""
    You are an Editorial Director planning a deep research report on "{topic}".
    We need to write exactly {num_chapters} DISTINCT, non-overlapping chapters based on the Knowledge Graph Triples below.

    TASK:
    Partition all the available triple IDs [T0, T1, T2...] into {num_chapters} thematic focuses.
    - Chapter 1: Cover foundational concepts, origins, core definitions, or early history.
    - Chapter 2+: Cover applications, developments, challenges, key features, or future implications.

    VERIFIED TRIPLES AVAILABLE:
    {kg_triples}

    OUTPUT FORMAT: Return strictly a JSON list of {num_chapters} strings describing the target focus and triple IDs for each chapter.
    EXAMPLE:
    [
      "Focus on Core Definitions, History, and Founding Context (Triples T0 to T8)",
      "Focus on Major Features, Applications, and Industry Use Cases (Triples T9 to T20)"
    ]
    """
    
    response = await call_ollama(prompt, temp=0.0)
    try:
        json_start = response.find("[")
        json_end = response.rfind("]") + 1
        if json_start != -1 and json_end != -1:
            outlines = safe_parse_json(response[json_start:json_end])
            if len(outlines) == num_chapters:
                return outlines
    except Exception as e:
        st.warning(f"Outline planner error: {e}. Using automatic slice strategy.")
        
    # Fallback: Partition triples evenly if LLM JSON fails
    step = max(1, len(edges) // num_chapters)
    fallback_outlines = []
    for i in range(num_chapters):
        start = i * step
        end = len(edges) if i == num_chapters - 1 else (i + 1) * step
        fallback_outlines.append(f"Focus specifically on Knowledge Graph Triples T{start} through T{end-1}")
    return fallback_outlines

# Stage 4B: Grounded Synthesis Engine
async def generate_chapter_from_graph(chapter_num: int, total_chapters: int, topic: str, chapter_focus: str, graph_data: dict, temp: float) -> str:
    """Synthesizes chapter content strictly grounded in its assigned thematic focus."""
    kg_triples = "\n".join(
        [f"[T{idx}] ({e.get('source')}) ──[{e.get('relation')}]──► ({e.get('target')})"
         for idx, e in enumerate(graph_data.get("edges", []))]
    )
    
    prompt = f"""
    You are an Investigative Knowledge Analyst writing Chapter {chapter_num} of {total_chapters} on "{topic}".

    ASSIGNED CHAPTER FOCUS:
    {chapter_focus}

    CRITICAL ACCURACY & DEDUPLICATION GUARDRAILS:
    1. Write EXCLUSIVELY about the assigned focus above.
    2. Do NOT write a general intro/summary if this is Chapter 2 or later. Do not repeat facts covered in earlier chapters.
    3. Every claim or sentence must explicitly cite its corresponding triple ID in brackets (e.g., [T1], [T2]).
    4. Write in a clear, well-structured, professional report style.

    VERIFIED KNOWLEDGE GRAPH TRIPLES:
    {kg_triples}
    """
    return await call_ollama(prompt, temp=temp, timeout=300)

# Graph Visualization UI
def visualize_knowledge_graph(graph_data: dict):
    net = Network(height="450px", width="100%", bgcolor="#111111", font_color="white")
    
    nodes_added = set()
    for node in graph_data.get("nodes", []):
        node_id = node.get("id")
        if node_id and node_id not in nodes_added:
            net.add_node(node_id, label=node_id, title=f"Type: {node.get('type', 'Unknown')}")
            nodes_added.add(node_id)
        
    for edge in graph_data.get("edges", []):
        src = edge.get("source")
        tgt = edge.get("target")
        rel = edge.get("relation", "related_to")
        if src in nodes_added and tgt in nodes_added:
            net.add_edge(src, tgt, title=rel, label=rel)
        
    html_path = "knowledge_graph.html"
    net.save_graph(html_path)
    
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=470)

# --- Main Async Core Pipeline ---
async def async_research_pipeline():
    if not topic.strip():
        st.error("Please provide a valid research topic.")
        return

    flux_pipe = None
    if enable_images:
        flux_pipe = load_local_flux_pipeline()

    status_box = st.status("🎬 Starting Zero-Hallucination Knowledge Engine Pipeline...", expanded=True)
    
    # 1. Discovery
    status_box.write("🔍 Querying web indices...")
    urls = await search_web(topic, max_links)
    
    if not urls:
        status_box.update(label="❌ Search Failed.", state="error")
        st.error("No relevant target sources found.")
        return
        
    status_box.write(f"🌐 Located {len(urls)} live target sources. Scraping content...")
    
    # 2. Scraping
    browser_config = BrowserConfig(headless=True, enable_stealth=True)
    prune_filter = PruningContentFilter(threshold=0.45, threshold_type="dynamic")
    md_generator = DefaultMarkdownGenerator(content_filter=prune_filter)
    
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=150,
        page_timeout=60000,
        remove_overlay_elements=True,
        markdown_generator=md_generator
    )
    raw_corpus = ""
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun_many(urls=urls, config=run_config)
        for idx, res in enumerate(results):
            # SAFE URL RETRIEVAL TO PREVENT INDEX ERROR
            scraped_url = getattr(res, 'url', None) or (urls[idx] if idx < len(urls) else f"Source #{idx+1}")
            
            if res.success and res.markdown:
                content = getattr(res.markdown, 'fit_markdown', res.markdown)
                if content and str(content).strip():
                    raw_corpus += f"\n{content}"
                    status_box.write(f"✅ Scraped Source #{idx+1}: {scraped_url}")
                    continue
            status_box.write(f"⚠️ Warning: Failed to scrape Source #{idx+1}: {scraped_url}")
                
    if not raw_corpus.strip():
        status_box.update(label="❌ Scraping Failed.", state="error")
        st.error("No valid text extracted from sources.")
        return

    # 3. Stage 1: Noise Filtering
    chunks = chunk_text(raw_corpus, chunk_size)
    status_box.write(f"🧹 Stage 1: Filtering off-topic web noise across {len(chunks)} segment(s)...")
    clean_chunks = await filter_relevant_chunks(chunks, topic)

    # 4. Stage 2: Fact Disambiguation
    status_box.write("✂️ Stage 2: Disambiguating claims into atomic facts...")
    all_disambiguated_facts = []
    progress_bar = st.progress(0, text="Disambiguating text...")
    for idx, chunk in enumerate(clean_chunks):
        progress_bar.progress((idx + 1) / len(clean_chunks), text=f"Processing segment {idx+1}/{len(clean_chunks)}...")
        facts = await disambiguate_facts(chunk)
        all_disambiguated_facts.extend(facts)
    progress_bar.empty()

    # 5. Stage 3A: Candidate Graph Construction
    status_box.write("🕸️ Stage 3A: Extracting candidate knowledge graph triples...")
    raw_graph_data = await build_knowledge_graph(all_disambiguated_facts)

    # 6. Stage 3B: Critic Loop Verification
    status_box.write("🕵️ Stage 3B: Running Qwen Fact-Critic Verification Loop...")
    verified_graph_data = await verify_triples_critic(raw_corpus, raw_graph_data)

    status_box.update(label="🎉 Knowledge Base Verified and Sanitized!", state="complete", expanded=False)

    # 7. Interactive Graph & Universal Grounded Synthesis
    st.subheader("🕸️ Verified Knowledge Graph Network (Post-Critic)")
    visualize_knowledge_graph(verified_graph_data)

    st.divider()
    st.markdown(f"# 📄 Verified Investigative Report: {topic}")
    st.divider()

    # Create distinct outlines for chapters
    status_box.write("📋 Planning thematic chapter outlines...")
    chapter_outlines = await create_chapter_outline(topic, num_chapters, verified_graph_data)

    for ch in range(1, num_chapters + 1):
        chapter_focus = chapter_outlines[ch - 1]
        st.markdown(f"## Chapter {ch}")
        st.caption(f"🎯 **Chapter Focus:** {chapter_focus}")

        with st.spinner(f"Synthesizing Chapter {ch} based on assigned outline..."):
            chapter_text = await generate_chapter_from_graph(
                chapter_num=ch,
                total_chapters=num_chapters,
                topic=topic,
                chapter_focus=chapter_focus,
                graph_data=verified_graph_data,
                temp=temperature
            )
            st.markdown(chapter_text)

        if enable_images and flux_pipe:
            with st.spinner(f"🎨 Generating Local FLUX.1 Image for Chapter {ch}..."):
                prompt_builder = f"Generate a detailed photorealistic image prompt about {topic} based on this summary: {chapter_text[:200]}. Output only prompt text."
                flux_prompt = await call_ollama(prompt_builder, temp=0.3)
                img = generate_local_flux_image(flux_pipe, flux_prompt.strip())
                if img:
                    st.image(img, caption=f"Figure {ch}: {flux_prompt.strip()}", use_container_width=True)

        st.divider()

    with st.expander("Show Raw Knowledge Graph JSON Data"):
        st.json(verified_graph_data)

# --- Event Loop Wrapper ---
def run_app():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        asyncio.ensure_future(async_research_pipeline())
    else:
        asyncio.run(async_research_pipeline())

if st.button("🚀 Execute Autonomous Research & Generation Campaign", type="primary"):
    run_app()