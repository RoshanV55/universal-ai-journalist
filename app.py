import os
import json
import ast
import re
import asyncio
import requests
import torch
import nest_asyncio
import streamlit as st
from PIL import Image
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import T5EncoderModel, BitsAndBytesConfig
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, CacheMode
from ddgs import DDGS
from pyvis.network import Network
import streamlit.components.v1 as components


def extract_and_sanitize_json(raw_response: str) -> dict | list:
    """Extracts, cleans, and parses JSON output from LLM responses."""
    if not raw_response or not isinstance(raw_response, str):
        return {}
        
    # Remove markdown codeblock fences if present
    cleaned = re.sub(r'```[a-zA-Z]*', '', raw_response)
    cleaned = cleaned.replace('```', '').strip()
    
    # Locate boundaries
    json_start = min([pos for pos in [cleaned.find('{'), cleaned.find('[')] if pos != -1], default=-1)
    json_end = max([cleaned.rfind('}'), cleaned.rfind(']')], default=-1) + 1
    
    if json_start != -1 and json_end != -1:
        json_str = cleaned[json_start:json_end]
        
        # Repair common unescaped key quote typos from local LLMs (e.g., {"id: "val"} -> {"id": "val"})
        json_str = re.sub(r'([{\[,])\s*"([^"]*?):', r'\1"\2":', json_str)
        
        return safe_parse_json(json_str)
        
    return {}

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
    return {}

# --- Environment Configuration ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- Model Configurations ---
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# --- Initialize Execution Lock State ---
if "is_running" not in st.session_state:
    st.session_state.is_running = False

# --- Streamlit Page Setup ---
st.set_page_config(page_title="AI Investigative Journalist & KG Generator", page_icon="📰", layout="wide")

st.title("📰 Universal AI Investigative Journalist & Knowledge Graph Engine")
st.caption("Deep research via Crawl4AI, Custom Ingestion, Anti-Bot Engine, Fact Disambiguation, and Knowledge Graphs.")

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

# --- UI Sidebar Setup ---
st.sidebar.header("Research Configurations")

# Data Ingestion Mode Selection
scrape_mode = st.sidebar.radio(
    "Data Source Selection Mode:",
    options=["Web Search + Custom Links", "Custom Links Only"],
    help="Choose whether to search the web, crawl specific custom links, or combine both."
)

custom_urls_input = st.sidebar.text_area(
    "Custom Website URLs (one per line):",
    placeholder="https://example.com/article1\nhttps://example.com/article2",
    height=120,
    help="Add websites (non-video) you wish to extract data from."
)

enable_stealth_browser = st.sidebar.checkbox(
    "Enable Anti-Bot Stealth Engine",
    value=True,
    help="Uses stealth browser signatures and anti-detection evasion techniques."
)

interactive_fallback = st.sidebar.checkbox(
    "Headed Mode (Visual Browser)",
    value=False,
    help="Launches a visible browser window if websites require manual verification."
)

max_links = st.sidebar.slider("Maximum Web Links to Explore", min_value=5, max_value=12, value=5, disabled=st.session_state.is_running)
chunk_size = st.sidebar.slider("Extraction Window Size (Words)", min_value=800, max_value=2500, value=1500, disabled=st.session_state.is_running)
temperature = st.sidebar.slider("Model Temperature (Creativity Control)", min_value=0.0, max_value=1.0, value=0.1, disabled=st.session_state.is_running)
num_chapters = st.sidebar.slider("Number of Chapters to Generate", min_value=1, max_value=6, value=3, disabled=st.session_state.is_running)
enable_images = st.sidebar.checkbox("Generate Local FLUX.1 Images for Chapters", value=False,disabled=st.session_state.is_running)

# Main Topic Input
topic = st.text_area(
    "Enter Deep Research Target / Investigative Topic:",
    value="",
    height=70,
    disabled=st.session_state.is_running,
)
# --- URL Parsing Helper ---
def parse_and_clean_urls(raw_text: str) -> list[str]:
    """Extracts valid web URLs from text split by newlines, commas, spaces, or semicolons."""
    raw_urls = [
        u.strip()
        for u in re.split(r'[\n,\s;]+', raw_text)
        if u.strip()
    ]
    
    valid_urls = []
    blocked_domains = ["youtube.com", "youtu.be", "vimeo.com", "tiktok.com", "dailymotion.com"]
    
    for url in raw_urls:
        if not (url.startswith("http://") or url.startswith("https://")):
            url = "https://" + url
            
        if not any(domain in url.lower() for domain in blocked_domains):
            # Target GitHub README directly if a raw repo link is provided
            if "github.com" in url.lower() and not ("raw.githubusercontent.com" in url or "blob/" in url):
                clean_url = url.rstrip("/")
                # Convert github.com/owner/repo to direct raw README source
                url = f"{clean_url}/raw/main/README.md"

            valid_urls.append(url)
            
    return list(dict.fromkeys(valid_urls))

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

# --- Pipeline Modules ---

# Stage 1: Noise & Ad Filter
async def filter_relevant_chunks(chunks: list[str], topic: str) -> list[str]:
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
    facts_block = "\n".join(f"- {f}" for f in disambiguated_facts)
    
    prompt = f"""
    You are an expert Knowledge Graph Extraction Engine.
    Extract Subject-Predicate-Object triples from the provided factual statements.

    OUTPUT FORMAT: Return strictly valid JSON in this exact structure without markdown tags:
    {{
      "nodes": [
        {{"id": "Entity Name", "type": "Person/Organization/Location/Concept/Event/Metric/Date/Technology"}}
      ],
      "edges": [
        {{"source": "Entity Name", "target": "Target Entity", "relation": "relationship_predicate"}}
      ]
    }}

    CRITICAL RULES:
    1. Output strictly valid raw JSON only. Do not wrap in markdown syntax or block quotes.
    2. Ensure every source and target entity in "edges" exists in "nodes".
    3. Escape all embedded quotes inside node and edge strings properly.

    FACTS TO PROCESS:
    {facts_block}
    """
    
    response = await call_ollama(prompt, temp=0.0)
    graph_data = extract_and_sanitize_json(response)
    
    if isinstance(graph_data, dict) and "nodes" in graph_data and "edges" in graph_data:
        return graph_data
    else:
        st.warning("Could not structure valid graph data from LLM response.")
        return {"nodes": [], "edges": []}

# Stage 3B: Critic Loop Verification Agent
async def verify_triples_critic(raw_text_corpus: str, unverified_graph: dict) -> dict:
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
    3. Output STRICTLY a raw JSON array of objects with "id" and "status" (1 for PASS, 0 for FAIL).

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
    verdict_list = extract_and_sanitize_json(response)

    if isinstance(verdict_list, list) and len(verdict_list) > 0:
        passed_ids = {item["id"] for item in verdict_list if isinstance(item, dict) and item.get("status") == 1}
        verified_edges = [edge for idx, edge in enumerate(edges) if idx in passed_ids]
        
        active_node_names = set()
        for edge in verified_edges:
            active_node_names.add(edge.get("source"))
            active_node_names.add(edge.get("target"))

        verified_nodes = [n for n in nodes if n.get("id") in active_node_names]

        return {"nodes": verified_nodes, "edges": verified_edges}

    st.warning("Verification parser error. Retaining raw graph data.")
    return unverified_graph

# Stage 4A: Universal Outline Planner Agent
async def create_chapter_outline(topic: str, num_chapters: int, graph_data: dict) -> list[str]:
    edges = graph_data.get("edges", [])
    if not edges:
        return [f"General Overview and Findings on {topic}" for _ in range(num_chapters)]

    kg_triples = "\n".join(
        [f"[T{idx}] ({e.get('source')}) ──[{e.get('relation')}]──► ({e.get('target')})"
         for idx, e in enumerate(edges)]
    )
    
    prompt = f"""
    You are an Editorial Director planning a deep research report on "{topic}".
    We need to write exactly {num_chapters} DISTINCT, non-overlapping chapters based on the Knowledge Graph Triples below.

    TASK:
    Partition all the available triple IDs [T0, T1, T2...] into {num_chapters} thematic focuses.

    VERIFIED TRIPLES AVAILABLE:
    {kg_triples}

    OUTPUT FORMAT: Return strictly a raw JSON list of {num_chapters} strings describing the target focus and triple IDs for each chapter.
    EXAMPLE:
    [
      "Focus on Core Concepts and Architecture (Triples T0 to T8)",
      "Focus on Features, Workflows, and Performance (Triples T9 to T20)"
    ]
    """
    
    response = await call_ollama(prompt, temp=0.0)
    outlines = extract_and_sanitize_json(response)
    
    if isinstance(outlines, list) and len(outlines) == num_chapters:
        return outlines
        
    step = max(1, len(edges) // num_chapters)
    fallback_outlines = []
    for i in range(num_chapters):
        start = i * step
        end = len(edges) if i == num_chapters - 1 else (i + 1) * step
        fallback_outlines.append(f"Focus specifically on Knowledge Graph Triples T{start} through T{end-1}")
    return fallback_outlines

# Stage 4B: Grounded Synthesis Engine
async def generate_chapter_from_graph(chapter_num: int, total_chapters: int, topic: str, chapter_focus: str, graph_data: dict, raw_facts: list[str], temp: float) -> str:
    edges = graph_data.get("edges", [])
    
    if edges:
        kg_triples = "\n".join(
            [f"[T{idx}] ({e.get('source')}) ──[{e.get('relation')}]──► ({e.get('target')})"
             for idx, e in enumerate(edges)]
        )
        context_block = f"VERIFIED KNOWLEDGE GRAPH TRIPLES:\n{kg_triples}"
    else:
        facts_block = "\n".join([f"- {fact}" for fact in raw_facts[:35]])
        context_block = f"VERIFIED RAW FACTS:\n{facts_block}"

    prompt = f"""
    You are an Investigative Knowledge Analyst writing Chapter {chapter_num} of {total_chapters} on "{topic}".

    ASSIGNED CHAPTER FOCUS:
    {chapter_focus}

    CRITICAL ACCURACY & GROUNDING RULES:
    1. Write EXCLUSIVELY about the assigned topic using ONLY the factual context provided below.
    2. Do NOT invent external tools, libraries, or concepts not present in the provided context.
    3. If citing Knowledge Graph Triples, cite their corresponding IDs in brackets (e.g., [T1], [T2]).
    4. Write in a clean, professional, structured report format with subheadings.

    {context_block}
    """
    return await call_ollama(prompt, temp=temp, timeout=300)

# Graph Visualization UI
def visualize_knowledge_graph(graph_data: dict):
    if not graph_data.get("nodes"):
        st.info("No network nodes available to render graph.")
        return

    net = Network(height="450px", width="100%", bgcolor="#111111", font_color="white")
    
    nodes_added = set()
    for node in graph_data.get("nodes", []):
        node_id = node.get("id")
        if node_id and node_id not in nodes_added:
            net.add_node(node_id, label=str(node_id), title=f"Type: {node.get('type', 'Unknown')}")
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
    
    components.html(html_content, height=470, scrolling=True)

# --- Main Async Core Pipeline ---
async def async_research_pipeline():
    user_urls = parse_and_clean_urls(custom_urls_input)
    target_urls = []

    status_box = st.status("🎬 Starting Zero-Hallucination Knowledge Engine Pipeline...", expanded=True)
    
    if scrape_mode == "Custom Links Only":
        if not user_urls:
            status_box.update(label="❌ Error: No valid custom URLs provided.", state="error")
            st.error("Please enter at least one valid web URL in the sidebar text box.")
            return
        target_urls = user_urls
        status_box.write(f"📌 Using {len(target_urls)} custom user-provided URL(s).")
    else:
        if not topic.strip() and not user_urls:
            status_box.update(label="❌ Missing Inputs.", state="error")
            st.error("Please provide either a research topic or at least one custom URL.")
            return

        searched_urls = []
        if topic.strip():
            status_box.write("🔍 Querying web indices...")
            searched_urls = await search_web(topic, max_links)
            
        combined = user_urls + searched_urls
        target_urls = list(dict.fromkeys(combined))
        
        if not target_urls:
            status_box.update(label="❌ Search Failed.", state="error")
            st.error("No target sources found from search or inputs.")
            return
        status_box.write(f"🌐 Targets established: {len(target_urls)} link(s) ({len(user_urls)} custom, {len(searched_urls)} searched).")

    flux_pipe = None
    if enable_images:
        flux_pipe = load_local_flux_pipeline()

    browser_config = BrowserConfig(
        headless=not interactive_fallback,
        user_agent_mode="random",
        headers={"Accept-Language": "en-US,en;q=0.9"}
    )
    
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=20,
        page_timeout=60000,
        remove_overlay_elements=True,
        css_selector="article.markdown-body, main, body",  # Targets GitHub README markdown container specifically
        excluded_tags=["nav", "header", "footer", "aside"],
        delay_before_return_html=3.0
    )
    raw_corpus = ""
    
    status_box.write(f"🕷️ Navigating and extracting content from {len(target_urls)} links...")
    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun_many(urls=target_urls, config=run_config)
        for idx, res in enumerate(results):
            scraped_url = getattr(res, 'url', None) or (target_urls[idx] if idx < len(target_urls) else f"Source #{idx+1}")
            
            if res.success:
                extracted_content = ""
                
                if hasattr(res, 'markdown_v2') and res.markdown_v2:
                    md_v2 = res.markdown_v2
                    extracted_content = getattr(md_v2, 'raw_markdown', getattr(md_v2, 'fit_markdown', str(md_v2)))
                elif hasattr(res, 'markdown') and res.markdown:
                    md = res.markdown
                    if isinstance(md, str):
                        extracted_content = md
                    elif hasattr(md, 'raw_markdown') and md.raw_markdown:
                        extracted_content = md.raw_markdown
                    elif hasattr(md, 'fit_markdown') and md.fit_markdown:
                        extracted_content = md.fit_markdown
                    else:
                        extracted_content = str(md)
                
                if not extracted_content.strip() and hasattr(res, 'cleaned_html') and res.cleaned_html:
                    extracted_content = res.cleaned_html
                elif not extracted_content.strip() and hasattr(res, 'html') and res.html:
                    extracted_content = res.html

                if extracted_content and extracted_content.strip():
                    raw_corpus += f"\n\n--- SOURCE: {scraped_url} ---\n{extracted_content}"
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
    status_box.write("🕵️ Stage 3B: Running Fact-Critic Verification Loop...")
    verified_graph_data = await verify_triples_critic(raw_corpus, raw_graph_data)

    status_box.update(label="🎉 Knowledge Base Verified and Sanitized!", state="complete", expanded=False)

    # 7. Interactive Graph & Universal Grounded Synthesis
    st.subheader("🕸️ Verified Knowledge Graph Network (Post-Critic)")
    visualize_knowledge_graph(verified_graph_data)

    st.divider()
    st.markdown(f"# 📄 Verified Investigative Report: {topic if topic else 'Custom Ingestion Analysis'}")
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
                raw_facts=all_disambiguated_facts,
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

# --- Execution Entrypoint ---
if st.button("🚀 Execute Autonomous Research & Generation Campaign", type="primary", disabled=st.session_state.is_running):
    st.session_state.is_running = True
    st.rerun()

# Run the pipeline when locked state is active
if st.session_state.is_running:
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import nest_asyncio
            nest_asyncio.apply(loop)
            loop.run_until_complete(async_research_pipeline())
        else:
            asyncio.run(async_research_pipeline())
    except Exception as e:
        st.error(f"Execution Error encountered during pipeline run: {e}")
    finally:
        # Re-enable the UI components once the pipeline completes or throws an exception
        st.session_state.is_running = False
        st.rerun()