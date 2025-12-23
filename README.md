# Zesty RAG Assignment

A production-ready RAG (Retrieval-Augmented Generation) system for analyzing insurance regulatory PDFs with mixed structured and unstructured content.

## Overview

This system implements a hybrid RAG architecture capable of answering complex questions requiring both semantic understanding and precise quantitative data extraction from structured tables. It processes 40+ PDF documents and handles three types of queries:

1. **List Queries**: Comprehensive enumeration (e.g., "List all rating plan rules")
2. **Comparison Queries**: Geographic reasoning with territory-based comparisons
3. **Calculation Queries**: Multi-step calculations requiring values from multiple tables

## Architecture

### Dual-Index Strategy
- **DOC Index**: BigQuery Vector Store for semantic retrieval of text chunks and table summaries
- **FACTS Store**: BigQuery tables for precise structured data queries

### Hybrid Retrieval
- Semantic search (vector similarity)
- Value-based table search
- Column-based table search
- Full table retrieval (instead of SQL generation)

### Key Features
- Handles inconsistent table schemas (generic headers, missing headers)
- Detects and merges tables split across pages
- Dynamic table discovery (no hardcoded table names)
- Page/document-level expansion for comprehensive context
- Parallel processing for performance
- Comprehensive evaluation framework

## Project Structure

```
.
├── src/
│   ├── parsing/          # PDF parsing and extraction
│   ├── indexing/         # Index creation (vector store, BigQuery)
│   ├── retrieval/        # RAG agent and retrieval logic
│   └── evaluation/      # Evaluation framework
├── artifacts/            # Questions, results, and evaluation reports
├── MD_docs/              # Architecture and design documentation
├── debug_scripts/        # Debugging and investigation scripts
└── test_results/        # Test outputs
```

## Setup

### Prerequisites
- Python 3.10+
- Google Cloud Project with BigQuery and Vertex AI enabled
- Service account with appropriate permissions

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Create a `.env` file:
```
GOOGLE_CLOUD_PROJECT=your-project-id
```

## Usage

### Build Indices

```bash
python3 build_indices.py
```

### Query RAG System

```bash
# Interactive mode
python3 query_rag.py

# Single query
python3 query_rag.py "Your question here"

# Batch queries from CSV
python3 query_rag.py --file artifacts/questions.csv
```

### Evaluation

```bash
# Run evaluation on existing results
python3 src/evaluation/evaluate.py
```

## Development Journey

See [RAG_DEVELOPMENT_JOURNEY.md](RAG_DEVELOPMENT_JOURNEY.md) for detailed documentation of:
- Thought process and reasoning
- Development phases
- Key technical decisions
- Lessons learned

## Results

- **EF_1 (List)**: 74.3% coverage (26/35 rules found)
- **EF_2 (Comparison)**: 100% accuracy (all facts correct)
- **EF_3 (Calculation)**: 100% accuracy (correct calculation)

## Documentation

- [RAG Development Journey](RAG_DEVELOPMENT_JOURNEY.md) - Complete development story
- [Unified Architecture](MD_docs/unified_architecture.md) - Architecture design
- [Evaluation Framework](src/evaluation/evaluate.py) - Evaluation metrics and methods

## License

This project is part of a technical assignment.
