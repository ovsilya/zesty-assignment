# Unified SOTA RAG Architecture - Solution Summary

## Overview

This document provides a high-level summary of the unified RAG architecture solution that combines the best approaches from three state-of-the-art designs (Gemini, GPT, and Grok architectures) to create a production-ready system for handling complex PDF documents with mixed structured and unstructured content.

---

## Problem Statement

**Challenge**: Classic RAG (Pinecone, simple chunking) underperforms on:
- 40 PDF documents (2-250 pages)
- Large, complex tables with varying structures
- Mixed document quality (digital + scanned)
- Inventory-like list tables
- Need for precise quantitative queries AND semantic understanding

**Goal**: Design and implement a SOTA RAG solution that outperforms classic approaches.

---

## Solution: Unified Architecture

### Core Innovation: Dual-Index Strategy

Instead of a single vector store, we use:

1. **DOC Index** (Vector Store)
   - Text chunks + table summaries
   - Semantic retrieval via embeddings
   - Handles "what is", "explain", "describe" queries

2. **FACTS Store** (BigQuery)
   - Full structured tables
   - Precise SQL queries
   - Handles "calculate", "filter", "list from table" queries

### Key Components

```
PDF Documents
    ↓
[Unstructured.io] → Parse & Extract (text + tables)
    ↓
    ├─→ Text Chunks → [Embeddings] → DOC Index (Vector Store)
    └─→ Tables → [LLM Summary] → DOC Index
              └─→ [Full Table] → FACTS Store (BigQuery)
    ↓
[Agentic Router] → Classify Query
    ↓
    ├─→ Semantic? → DOC Index (Vector Search) → Re-rank
    └─→ Quantitative? → FACTS Store (SQL Query)
    ↓
[LLM] → Generate Answer with Sources
    ↓
[RAGAS] → Evaluate vs. Golden Dataset
```

---

## Tool Stack (Common Across All Three Architectures)

### Core Tools
- **Unstructured.io**: PDF parsing with table extraction
- **LangChain**: Orchestration and agent framework
- **BigQuery**: Unified storage (vector + structured)
- **Vertex AI**: Embeddings and LLMs (Gemini)
- **FlashRank**: Lightweight reranking
- **RAGAS**: Evaluation framework

### Alternative Options
- **Pinecone/ChromaDB**: Vector store (if not using GCP)
- **OpenAI/Claude**: LLMs (alternative to Gemini)
- **Cohere**: Reranking API (alternative to FlashRank)

---

## Implementation Phases

### Phase 1: Document Processing (Week 1)
- Parse 40 PDFs with Unstructured.io
- Extract and normalize tables
- Generate table summaries with LLM
- Handle both digital and scanned PDFs

### Phase 2: Indexing (Week 1-2)
- Build DOC Index (vector store) with embeddings
- Load tables into FACTS Store (BigQuery)
- Enrich with metadata (page numbers, document IDs)

### Phase 3: RAG Agent (Week 2)
- Implement agentic router (query classification)
- Create hybrid retrieval (vector + SQL)
- Integrate reranking layer
- Test on sample queries

### Phase 4: Evaluation (Week 2-3)
- Set up RAGAS evaluation pipeline
- Run on golden dataset (questions.csv)
- Compare vs. baseline (classic RAG)
- Generate metrics report

### Phase 5: Optimization (Week 3)
- Fine-tune retrieval parameters
- Optimize chunk sizes
- Improve SQL query generation
- Iterate based on results

---

## Expected Performance Improvements

Based on research from all three architectures:

| Metric | Classic RAG | Unified Architecture | Improvement |
|--------|-------------|---------------------|-------------|
| Retrieval Accuracy | Baseline | +10-15% | Hybrid search |
| Table Query Precision | Baseline | +7-20% | Structured storage |
| Answer Faithfulness | Baseline | Higher | Reranking + dual-index |
| Complex Query Handling | Limited | Multi-step reasoning | Agentic routing |

---

## Key Differentiators

### 1. **Intelligent Query Routing**
- LLM classifies queries as semantic, quantitative, or hybrid
- Routes to appropriate retrieval method automatically
- Can combine both methods for complex queries

### 2. **Structure Preservation**
- Tables stored in structured format (not flattened)
- Enables precise SQL queries
- Summaries enable semantic discovery

### 3. **Hybrid Retrieval**
- Dense vector search for semantic understanding
- SQL queries for precise data extraction
- Optional: BM25 for keyword matching

### 4. **Production-Ready Tools**
- BigQuery: Scalable, serverless, unified infrastructure
- LangChain: Industry-standard orchestration
- RAGAS: Comprehensive evaluation framework

---

## File Structure

```
Zesty/
├── unified_architecture.md      # Detailed architecture design
├── implementation_guide.md       # Step-by-step implementation code
├── architecture_comparison.md    # Tool overlap analysis
├── SOLUTION_SUMMARY.md          # This file
│
├── gemini_arch.txt              # Original architecture 1
├── gpt_arch.txt                 # Original architecture 2
├── grok_arch.txt                # Original architecture 3
│
└── artifacts/
    ├── questions.csv            # Golden dataset
    ├── README.md                # Evaluation instructions
    └── [PDF documents]          # 40 PDF files
```

---

## Quick Start

### 1. Review Architecture
- Read `unified_architecture.md` for design details
- Review `architecture_comparison.md` for tool rationale

### 2. Set Up Environment
```bash
# Install dependencies (see implementation_guide.md)
pip install unstructured[all-docs] langchain langchain-google-vertexai ...
```

### 3. Run Pipeline
```bash
# Process PDFs and build indexes
python main.py

# Run evaluation
python src/evaluation/evaluate.py
```

### 4. Review Results
- Check evaluation report (HTML)
- Compare metrics vs. baseline
- Iterate on improvements

---

## Evaluation Metrics

The solution will be evaluated using RAGAS on:

1. **Retrieval Metrics**
   - Context Precision: Relevant docs / Total retrieved
   - Context Recall: Relevant docs / Total relevant in corpus

2. **Generation Metrics**
   - Faithfulness: Factual consistency with context
   - Answer Relevance: Extent answer addresses question

3. **Comparison**
   - Baseline: Classic RAG (Pinecone, simple chunking)
   - Target: >10-15% improvement on retrieval accuracy
   - Target: >7-20% improvement on table query precision

---

## Why This Solution Works

### Addresses Classic RAG Limitations

| Classic RAG Issue | Unified Solution |
|-------------------|------------------|
| ❌ Naive chunking breaks tables | ✅ Separate table extraction |
| ❌ Flattened table data | ✅ Structured BigQuery storage |
| ❌ Single retrieval method | ✅ Hybrid (vector + SQL) |
| ❌ No query routing | ✅ Agentic LLM-based routing |
| ❌ Poor reranking | ✅ Cross-encoder reranking |
| ❌ Limited evaluation | ✅ Comprehensive RAGAS metrics |

### Based on Proven Tools

- **Unstructured.io**: Industry standard for PDF parsing
- **LangChain**: Most popular RAG framework
- **BigQuery**: Google's scalable data warehouse
- **RAGAS**: Leading evaluation framework

### Combines Best Practices

- **From Gemini**: Hybrid search, FlashRank, comprehensive evaluation
- **From GPT**: MCP connectors, multi-modal LLMs, CoT prompting
- **From Grok**: Unified infrastructure, LangGraph agents, production tools

---

## Next Steps for Implementation

1. **Prototype** (Days 1-3)
   - Set up GCP project and BigQuery
   - Process 2-3 sample PDFs
   - Test table extraction quality

2. **Full Pipeline** (Days 4-7)
   - Process all 40 PDFs
   - Build both indexes
   - Implement RAG agent

3. **Evaluation** (Days 8-10)
   - Run on golden dataset
   - Compare vs. baseline
   - Generate report

4. **Optimization** (Days 11-14)
   - Fine-tune parameters
   - Improve SQL generation
   - Iterate on results

---

## Success Criteria

✅ **Functional Requirements**
- Handles all 40 PDFs (2-250 pages)
- Extracts tables accurately
- Answers semantic and quantitative queries
- Provides source attribution

✅ **Performance Requirements**
- >10% improvement in retrieval accuracy vs. baseline
- >7% improvement in table query precision
- Faithfulness score >0.85
- Answer relevance score >0.80

✅ **Quality Requirements**
- Handles mixed document quality
- Preserves table structure
- Supports multi-step reasoning
- Comprehensive evaluation metrics

---

## Conclusion

This unified architecture successfully combines the strengths of three SOTA RAG designs into a single, production-ready solution. By focusing on tools and patterns that appear across multiple architectures, we ensure:

- **Robustness**: Proven, industry-standard tools
- **Scalability**: BigQuery handles petabyte-scale
- **Flexibility**: Supports multiple deployment scenarios
- **Evaluability**: Comprehensive metrics via RAGAS

The solution addresses all limitations of classic RAG while maintaining consistency with best practices from leading architectures.

---

## References

- **Unified Architecture**: `unified_architecture.md`
- **Implementation Guide**: `implementation_guide.md`
- **Tool Comparison**: `architecture_comparison.md`
- **Original Designs**: `gemini_arch.txt`, `gpt_arch.txt`, `grok_arch.txt`

