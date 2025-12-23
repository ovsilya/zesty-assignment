# Unified SOTA RAG Architecture for Complex PDF Analysis
## Combining Best Practices from Gemini, GPT, and Grok Architectures

### Executive Summary

This architecture synthesizes the overlapping approaches from three state-of-the-art RAG designs to create a unified solution for handling 40 PDF documents with mixed structured (tables) and unstructured (text) content. The solution leverages tools and techniques that appear across all three architectures, ensuring a robust, production-ready system.

---

## Core Architecture Principles

### 1. **Dual-Index Strategy** (Common to All Three)
- **DOC Index**: Vector store for semantic retrieval of text and table summaries
- **FACTS Store**: Structured storage (BigQuery) for precise tabular queries via SQL

### 2. **Hybrid Retrieval** (Common to All Three)
- Dense vector search for semantic queries
- Structured/SQL queries for quantitative/precise lookups
- Agentic routing to intelligently select retrieval method

### 3. **Advanced Parsing Pipeline** (Common to All Three)
- Unstructured.io for document partitioning
- Vision models for scanned/low-quality PDFs
- Separate handling of text vs. tables

### 4. **Re-ranking Layer** (Common to All Three)
- Cross-encoder models to improve retrieval precision

### 5. **Evaluation Framework** (Common to All Three)
- RAGAS for comprehensive metrics
- Golden dataset comparison

---

## Unified Architecture Components

### Phase 1: Document Processing and Parsing

#### 1.1 Intelligent PDF Parsing

**Tools (Common Across All Three):**
- **Unstructured.io** with `hi_res` strategy for partitioning documents
- **LangChain/LlamaIndex** for orchestration
- **PaddleOCR** or **Gemini Vision API** for scanned documents (fallback)

**Process:**
```python
# Pseudo-code structure
1. For each PDF:
   a. Detect if digital or scanned (using Unstructured.io)
   b. If digital: Use Unstructured.io with infer_table_structure=True
   c. If scanned: Use OCR (PaddleOCR/Gemini Vision) + layout detection
   d. Extract elements: {text_blocks, tables, figures, metadata}
   e. Output: JSONL structure with page numbers, element types, coordinates
```

**Key Features:**
- Preserves table structure (not flattened)
- Maintains document hierarchy (sections, headers)
- Handles mixed-quality PDFs (2-250 pages)
- Extracts metadata (page numbers, document IDs)

#### 1.2 Table Extraction and Normalization

**Process:**
```python
For each extracted table:
  1. Convert to structured format:
     - Pandas DataFrame (for cleaning/normalization)
     - Preserve headers, data types, relationships
  2. Generate natural language summary:
     - Use LLM (Gemini/GPT-4) via LangChain
     - Summary format: "This table lists [category] by [dimension] with columns [X, Y, Z]"
  3. Store in dual format:
     - Summary → goes to DOC Index (vector store)
     - Full table → goes to FACTS Store (BigQuery)
```

**Tools:**
- **Pandas** for DataFrame operations
- **LangChain** for LLM-based summarization
- **BigQuery** for structured storage

---

### Phase 2: Dual-Index Creation

#### 2.1 DOC Index (Vector Store for Semantic Retrieval)

**Components:**
- **Text Chunks**: Recursive character splitting with overlap (chunk_size=1000, overlap=100)
- **Table Summaries**: LLM-generated summaries from Phase 1.2
- **Embeddings**: 
  - Primary: **Vertex AI text-embedding-005** (from Grok arch) OR
  - Alternative: **OpenAI text-embedding-3-large** (from GPT arch)
- **Storage**: 
  - **BigQuery Vector Store** (from Grok arch) - preferred for unified infrastructure
  - OR **Pinecone/ChromaDB** (from Gemini arch) if not using GCP

**Metadata Enrichment:**
- Document ID, page number, element type (text/table), table_id (if applicable)
- Enables provenance tracking and source citation

#### 2.2 FACTS Store (Structured Storage for Precise Queries)

**Components:**
- **Storage**: **BigQuery** (common to all three architectures)
- **Format**: Relational tables (one dataset per document type or unified schema)
- **Schema**: 
  - Table: `pdf_id`, `table_id`, `page_number`, `columns`, `data` (as structured rows)
  - Or: Individual BigQuery tables per extracted table with proper schemas

**MCP Integration:**
- Use **BigQuery MCP connector** (from GPT arch) for agentic access
- Enables SQL query generation and execution via LLM

**Alternative (if not using GCP):**
- **DuckDB** (mentioned in Gemini and GPT archs) for local/lightweight SQL queries

---

### Phase 3: Hybrid Retrieval and Agentic Routing

#### 3.1 Query Understanding and Routing

**Agent Framework:**
- **LangChain Agents** or **LangGraph** (from Grok arch) for orchestration
- **Routing Logic**:
  ```python
  Query Classification (LLM-based):
    - If query contains: numbers, dates, "calculate", "filter", "list from table"
      → Route to FACTS Store (SQL query)
    - If query is semantic: "what is", "explain", "describe"
      → Route to DOC Index (vector search)
    - If query is complex (multi-step)
      → Use both in sequence
  ```

#### 3.2 Hybrid Search Implementation

**DOC Index Retrieval:**
- **Dense Vector Search**: Semantic similarity using embeddings
- **Sparse Search (Optional)**: BM25 for keyword matching (from Gemini arch)
- **Hybrid Combination**: Weighted fusion of dense + sparse scores

**FACTS Store Retrieval:**
- **SQL Query Generation**: LLM generates SQL from natural language query
- **Execution**: BigQuery MCP connector executes query
- **Result Formatting**: Convert SQL results to natural language context

**Tools:**
- **LangChain SQL Agent** for SQL generation and execution
- **BigQuery MCP** for structured queries
- **LangGraph** for multi-step agentic workflows

#### 3.3 Re-ranking Layer

**Model Options (from all three):**
- **FlashRank** (from Gemini arch) - lightweight, fast
- **bge-reranker-large** (from GPT arch) - high accuracy
- **Cohere Rerank** (from Grok arch) - production-ready API

**Process:**
```python
1. Initial retrieval: Top K candidates (K=20-50)
2. Re-ranking: Cross-encoder scores each candidate against query
3. Final selection: Top N (N=5-10) most relevant chunks
4. Context assembly: Combine re-ranked chunks for LLM
```

---

### Phase 4: Generation and Synthesis

#### 4.1 LLM Selection

**Options (from all three):**
- **Gemini 1.5 Pro** (from Grok arch) - multimodal, good for tables
- **GPT-4 Turbo** (from GPT arch) - strong reasoning
- **Claude 3 Opus** (from GPT arch) - excellent for complex synthesis

**Implementation:**
- Use **LangChain** for LLM integration
- **Chain-of-Thought (CoT) prompting** (from GPT arch):
  - Include source attribution: "Based on Table X from Document Y..."
  - Request step-by-step reasoning for calculations

#### 4.2 Context Assembly

**Format:**
```
Context Sources:
1. [Text Chunk] from Document X, Page Y: "..."
2. [Table Summary] from Document Z, Page W: "This table shows..."
3. [SQL Result] from Table ABC: "The filtered data shows..."

Question: [User Query]

Instructions:
- Synthesize information from all sources
- Cite sources explicitly
- For calculations, show step-by-step reasoning
```

---

### Phase 5: Evaluation Framework

#### 5.1 Metrics (Common to All Three)

**Retrieval Metrics:**
- **Context Precision**: Ratio of relevant retrieved docs to total retrieved
- **Context Recall**: Ratio of relevant retrieved docs to total relevant in corpus

**Generation Metrics:**
- **Faithfulness**: Factual consistency with provided context
- **Answer Relevance**: Extent to which answer addresses the question

**Additional Metrics:**
- **Hallucination Rate**: Information not in context
- **Answer Completeness**: Coverage of expected answer points

#### 5.2 Evaluation Tools

**Primary: RAGAS** (mentioned in Gemini and Grok archs)
- Open-source, comprehensive metrics
- LLM-as-judge for nuanced scoring
- Integration with golden dataset

**Secondary:**
- **DeepEval** (from Gemini arch) - unit-testing style
- **Braintrust** (from Grok arch) - production CI/CD integration

#### 5.3 Evaluation Process

```python
1. Load golden dataset (questions.csv format)
2. For each question:
   a. Run RAG pipeline
   b. Store: query, generated_answer, retrieved_contexts, sources
3. Evaluate with RAGAS:
   - Compare generated_answer vs. expected_output
   - Score retrieval quality (precision, recall)
   - Score generation quality (faithfulness, relevance)
4. Generate report:
   - Per-question scores
   - Aggregate metrics
   - Comparison vs. baseline (classic RAG)
```

---

## Implementation Stack (Unified)

### Core Libraries
```python
# Document Processing
unstructured[all-docs]  # PDF parsing, table extraction
langchain                # Orchestration, agents
llama-index              # Alternative parsing/retrieval
pandas                   # DataFrame operations
pyarrow                  # Parquet support

# OCR (for scanned PDFs)
paddleocr                # OCR fallback
# OR google-generativeai  # Gemini Vision API

# Storage and Retrieval
google-cloud-bigquery    # FACTS Store + Vector Store
# OR pinecone-client      # Alternative vector store
# OR chromadb             # Alternative vector store

# Embeddings
langchain-google-vertexai  # Vertex AI embeddings
# OR openai                # OpenAI embeddings

# LLMs
langchain-google-vertexai  # Gemini
# OR langchain-openai      # GPT-4
# OR langchain-anthropic   # Claude

# Re-ranking
flashrank                # Lightweight reranker
# OR cohere               # Cohere rerank API

# Evaluation
ragas                    # Primary evaluation framework
datasets                 # Dataset handling
```

### Infrastructure Requirements
- **GCP Project** with BigQuery enabled (preferred)
- **Vertex AI** API access (for embeddings/LLMs)
- **Service Account** with appropriate permissions
- **Alternative**: OpenAI API key, Pinecone account (if not using GCP)

---

## Architecture Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    PDF Documents (40 files)                 │
│            (2-250 pages, mixed quality, tables/text)         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Phase 1: Document Processing                    │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │ Unstructured.io  │────────▶│ Table Extraction │          │
│  │ (hi_res parsing) │         │ + Normalization  │          │
│  └──────────────────┘         └──────────────────┘          │
│         │                              │                    │
│         │                              │                    │
│         ▼                              ▼                    │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │ Text Chunks      │         │ LLM Summaries   │          │
│  │ (recursive split)│         │ (table context) │          │
│  └──────────────────┘         └──────────────────┘          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Phase 2: Dual-Index Creation                   │
│  ┌────────────────────────┐   ┌────────────────────────┐   │
│  │   DOC Index            │   │   FACTS Store          │   │
│  │   (Vector Store)       │   │   (BigQuery)           │   │
│  │                        │   │                        │   │
│  │ • Text chunks          │   │ • Full tables          │   │
│  │ • Table summaries      │   │ • Structured format    │   │
│  │ • Embeddings           │   │ • SQL-queryable        │   │
│  │ • Metadata             │   │ • MCP accessible        │   │
│  └────────────────────────┘   └────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Phase 3: Query Processing                      │
│  ┌────────────────────────────────────────────────────┐   │
│  │         Agentic Router (LangChain/LangGraph)        │   │
│  │  ┌──────────────┐              ┌──────────────┐     │   │
│  │  │ Semantic?    │─────────────▶│ DOC Index    │     │   │
│  │  │              │              │ (Vector)     │     │   │
│  │  └──────────────┘              └──────────────┘     │   │
│  │  ┌──────────────┐              ┌──────────────┐     │   │
│  │  │ Quantitative?│─────────────▶│ FACTS Store  │     │   │
│  │  │              │              │ (SQL Query)  │     │   │
│  │  └──────────────┘              └──────────────┘     │   │
│  └────────────────────────────────────────────────────┘   │
│                         │                                    │
│                         ▼                                    │
│              ┌──────────────────────┐                       │
│              │   Re-ranking Layer   │                       │
│              │ (FlashRank/Cohere)   │                       │
│              └──────────────────────┘                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Phase 4: Generation                            │
│  ┌────────────────────────────────────────────────────┐     │
│  │  LLM (Gemini/GPT-4/Claude)                        │     │
│  │  • Context assembly                                │     │
│  │  • Source attribution                              │     │
│  │  • CoT reasoning                                   │     │
│  └────────────────────────────────────────────────────┘     │
│                         │                                    │
│                         ▼                                    │
│              ┌──────────────────────┐                       │
│              │   Final Answer       │                       │
│              │   + Sources          │                       │
│              └──────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Phase 5: Evaluation                            │
│  ┌────────────────────────────────────────────────────┐     │
│  │  RAGAS Framework                                   │     │
│  │  • Context Precision/Recall                        │     │
│  │  • Faithfulness                                    │     │
│  │  • Answer Relevance                                │     │
│  │  • Comparison vs. Golden Dataset                   │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Advantages of Unified Architecture

1. **Handles Mixed Content**: Separate processing for text vs. tables preserves structure
2. **Dual Retrieval**: Semantic search for concepts, SQL for precise data
3. **Agentic Intelligence**: LLM routes queries to appropriate retrieval method
4. **Production-Ready**: Uses proven tools (Unstructured.io, LangChain, BigQuery)
5. **Scalable**: BigQuery handles petabyte-scale, supports real-time updates
6. **Evaluable**: Comprehensive metrics via RAGAS
7. **Robust**: Handles both digital and scanned PDFs with OCR fallback

---

## Implementation Phases

### Phase 1: Data Ingestion (Week 1)
- Set up GCP project and BigQuery
- Implement PDF parsing pipeline with Unstructured.io
- Extract and normalize tables
- Generate table summaries with LLM

### Phase 2: Indexing (Week 1-2)
- Create DOC Index (vector store) with embeddings
- Load tables into BigQuery FACTS Store
- Test retrieval on sample queries

### Phase 3: RAG Agent (Week 2)
- Implement agentic router with LangChain/LangGraph
- Create hybrid retrieval (vector + SQL)
- Integrate re-ranking layer
- Test on golden dataset questions

### Phase 4: Evaluation (Week 2-3)
- Set up RAGAS evaluation pipeline
- Run full evaluation on golden dataset
- Compare vs. baseline (classic RAG)
- Generate metrics report

### Phase 5: Optimization (Week 3)
- Fine-tune retrieval parameters
- Optimize chunk sizes and overlap
- Improve SQL query generation
- Iterate based on evaluation results

---

## Expected Performance Improvements

Based on research from all three architectures:

- **Retrieval Accuracy**: 10-15% improvement over classic RAG (hybrid search)
- **Table Query Precision**: 7-20% improvement (structured storage + SQL)
- **Answer Quality**: Higher faithfulness and relevance (re-ranking + dual-index)
- **Handling Complex Queries**: Multi-step reasoning via agentic routing

---

## Next Steps

1. **Prototype Setup**: Create Python environment with all dependencies
2. **Sample Processing**: Test pipeline on 2-3 sample PDFs
3. **Incremental Development**: Build and test each phase independently
4. **Evaluation Baseline**: Run classic RAG first to establish baseline metrics
5. **Iterative Improvement**: Refine based on evaluation results

---

## References to Original Architectures

- **Gemini Architecture**: Dual-index strategy, FlashRank reranking, BM25 hybrid search
- **GPT Architecture**: MCP connectors, ColBERTv2 reranking, multi-modal LLMs
- **Grok Architecture**: LlamaParse alternative, BigQuery Vector Store, LangGraph agents, RAGAS evaluation

This unified architecture synthesizes the best of all three approaches while maintaining consistency with tools that appear across all designs.

