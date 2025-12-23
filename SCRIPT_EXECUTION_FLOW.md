# Script Execution Flow

This document describes the execution flow of all scripts in the RAG system, detailing how they interact and their role in the complete pipeline.

## Overview

The RAG system consists of three main execution phases:

1. **Indexing Phase**: Build indices from PDF documents
2. **Query Phase**: Answer questions using the built indices
3. **Evaluation Phase**: Evaluate system performance

---

## Main Pipeline Scripts

### 1. `build_indices.py` - Indexing Pipeline

**Purpose**: Process PDFs and build both DOC Index (vector store) and FACTS Store (BigQuery tables).

**Execution Flow**:

```
1. Initialize Components
   ├─→ PDFParser (LlamaParse + Unstructured)
   ├─→ TableProcessor (LLM summarization)
   ├─→ VectorIndexBuilder (embeddings)
   └─→ FactsStoreBuilder (BigQuery)

2. For Each PDF (in artifacts/1/ and artifacts/2/):
   │
   ├─→ [1/5] Parse PDF
   │   ├─→ LlamaParse: Extract tables, charts, structured text
   │   ├─→ Unstructured: Extract text elements
   │   └─→ Returns: ParsedDocument
   │
   ├─→ [2/5] Save Extracted Tables as CSV
   │   ├─→ Convert table rows to DataFrame
   │   ├─→ Apply header detection logic
   │   └─→ Save to extracted_tables/
   │
   ├─→ [3/5] Process Tables
   │   ├─→ Convert to DataFrame
   │   ├─→ Normalize (remove empty rows/cols)
   │   ├─→ Generate LLM summary (Gemini 2.5 Flash)
   │   └─→ Returns: List[ProcessedTable]
   │
   ├─→ [4/5] Add to DOC Index (Vector Store)
   │   ├─→ Extract text chunks (split into 1000-char chunks)
   │   ├─→ Create documents from table summaries
   │   ├─→ Create documents from chart descriptions
   │   ├─→ Generate embeddings (Vertex AI text-embedding-005)
   │   └─→ Upload to BigQuery Vector Store (batched)
   │
   └─→ [5/5] Store FACTS Tables
       ├─→ Sanitize column names (BigQuery compliance)
       ├─→ Flatten DataFrames (handle MultiIndex)
       ├─→ Add metadata columns (source, table_id, page_number)
       └─→ Upload to BigQuery as individual tables

3. Update Processing Log
   └─→ Save to processing_log.json (tracks processed PDFs)
```

**Key Features**:
- **Idempotent**: Skips already-processed PDFs (unless `FORCE_REPROCESS=1`)
- **Incremental**: Can add new PDFs without rebuilding entire index
- **Error Handling**: Logs errors but continues processing other PDFs

**Dependencies**:
- `src.parsing.pdf_parser.PDFParser`
- `src.parsing.table_processor.TableProcessor`
- `src.indexing.vector_store.VectorIndexBuilder`
- `src.indexing.facts_store.FactsStoreBuilder`

**Outputs**:
- **DOC Index**: `{project_id}.rag_dataset.doc_index` (BigQuery Vector Store)
- **FACTS Store**: `{project_id}.rag_dataset.table_*` (individual BigQuery tables)
- **CSV Files**: `extracted_tables/` (for inspection/comparison)
- **Processing Log**: `processing_log.json`

**Usage**:
```bash
python3 build_indices.py
```

**Environment Variables**:
- `GOOGLE_CLOUD_PROJECT`: Required
- `LLAMA_CLOUD_API_KEY`: Required (for PDF parsing)
- `FORCE_REPROCESS`: Optional (set to "1", "true", or "yes" to reprocess all PDFs)

---

### 2. `query_rag.py` - Query Script

**Purpose**: Query the RAG system using pre-built indices.

**Execution Flow**:

```
1. Initialize RAG Agent
   ├─→ Load BigQuery Vector Store (DOC Index)
   ├─→ Initialize BigQuery Client (for FACTS Store)
   ├─→ Load LLM (Gemini 2.5 Pro)
   ├─→ Load prompt template (prompt_template_v2.txt)
   └─→ Initialize RetrievalLogic

2. Determine Query Mode
   │
   ├─→ Interactive Mode (no arguments)
   │   └─→ Loop: prompt user → answer → display
   │
   ├─→ Single Query (question as argument)
   │   └─→ Answer question → display result
   │
   └─→ Batch Mode (--file argument)
       ├─→ Read questions from CSV
       ├─→ Process each question
       └─→ Save results to CSV

3. For Each Question:
   │
   ├─→ Semantic Retrieval
   │   ├─→ Query vector store (k=25-40 depending on query type)
   │   ├─→ Rerank with FlashRank (if available)
   │   └─→ Extract top results
   │
   ├─→ Page/Document Expansion
   │   ├─→ Find all content from same pages
   │   └─→ Expand to related documents
   │
   ├─→ Table Retrieval
   │   ├─→ Extract table IDs from semantic results
   │   ├─→ Value-based table search (extract values from query)
   │   ├─→ Column-based table search
   │   ├─→ Keyword matching
   │   ├─→ Detect and merge split tables
   │   └─→ Fetch full tables from BigQuery (parallel)
   │
   ├─→ Context Assembly
   │   ├─→ Combine semantic chunks
   │   ├─→ Add full table content (Markdown format)
   │   └─→ Add text from table pages
   │
   └─→ Answer Generation
       ├─→ Format prompt with context
       ├─→ Generate answer (Gemini 2.5 Pro)
       └─→ Return answer with sources
```

**Key Features**:
- **Three Modes**: Interactive, single query, batch processing
- **Hybrid Retrieval**: Always uses both semantic and structured retrieval
- **Source Attribution**: Includes document names, page numbers, table IDs

**Dependencies**:
- `src.retrieval.rag_agent.RAGAgent`
- `src.retrieval.retrieval_logic.RetrievalLogic`
- `src.retrieval.utils` (utility functions)

**Outputs**:
- **Interactive**: Prints answer to console
- **Single Query**: Prints answer to console
- **Batch Mode**: Saves to `{input_file}_results.csv`

**Usage**:
```bash
# Interactive mode
python3 query_rag.py

# Single query
python3 query_rag.py "List all rating plan rules"

# Batch queries
python3 query_rag.py --file artifacts/questions.csv
```

**Environment Variables**:
- `GOOGLE_CLOUD_PROJECT`: Required

---

### 3. `src/evaluation/evaluate.py` - Evaluation Framework

**Purpose**: Evaluate RAG system performance against ground truth.

**Execution Flow**:

```
1. Initialize Components
   ├─→ RAGAgent (for getting contexts if needed)
   └─→ RAGEvaluator

2. Load Questions and Results
   ├─→ Read questions.csv (questions + expected outputs)
   └─→ Read questions_results.csv (generated answers, optional)

3. For Each Question:
   │
   ├─→ If results_csv exists (fast mode):
   │   ├─→ Use cached answer from CSV
   │   └─→ Skip context extraction (skip_contexts=True)
   │
   └─→ If results_csv doesn't exist:
       ├─→ Run RAG agent to get answer
       └─→ Extract contexts for evaluation

4. Calculate Metrics
   ├─→ Exact Match
   ├─→ Keyword Coverage
   ├─→ Number Match
   ├─→ Answer Completeness
   ├─→ Source Citation Quality
   └─→ Question-Specific Metrics
       ├─→ EF_1: Rules Coverage
       ├─→ EF_2: Fact Score
       └─→ EF_3: Calculation Correctness

5. Generate Report
   ├─→ Summary statistics
   ├─→ Per-question results
   └─→ Save to evaluation_report.txt

6. Optional: RAGAS Evaluation
   └─→ If user confirms and ragas installed:
       ├─→ Get contexts for all questions
       ├─→ Run RAGAS metrics
       └─→ Display scores
```

**Key Features**:
- **Fast Mode**: Uses existing results, skips context extraction
- **Comprehensive Metrics**: Multiple evaluation dimensions
- **Question-Specific**: Custom metrics for each question type
- **Optional RAGAS**: Advanced metrics if package installed

**Dependencies**:
- `src.retrieval.rag_agent.RAGAgent`
- `src.evaluation.evaluate.RAGEvaluator`
- `ragas` (optional, for advanced metrics)

**Inputs**:
- `artifacts/questions.csv`: Questions and expected outputs
- `results_evaluation/questions_results.csv`: Generated answers (optional)

**Outputs**:
- `results_evaluation/evaluation_results.csv`: Detailed metrics
- `results_evaluation/evaluation_report.txt`: Summary report

**Usage**:
```bash
# Run evaluation (uses existing results if available)
python3 src/evaluation/evaluate.py

# Or as module
python3 -m src.evaluation.evaluate
```

**Environment Variables**:
- `GOOGLE_CLOUD_PROJECT`: Required

---

## Complete Pipeline Flow

### Phase 1: Indexing (One-Time Setup)

```bash
# Step 1: Build indices from PDFs
python3 build_indices.py
```

**What happens**:
1. Processes all PDFs in `artifacts/1/` and `artifacts/2/`
2. Extracts text, tables, and charts
3. Builds DOC Index (vector store) in BigQuery
4. Stores FACTS tables in BigQuery
5. Saves extracted tables as CSV for inspection

**Output**: Complete indices ready for querying

---

### Phase 2: Querying (Ongoing Use)

```bash
# Option 1: Interactive queries
python3 query_rag.py

# Option 2: Single query
python3 query_rag.py "Your question here"

# Option 3: Batch queries
python3 query_rag.py --file artifacts/questions.csv
```

**What happens**:
1. Loads pre-built indices
2. Performs hybrid retrieval (semantic + structured)
3. Generates answer using LLM
4. Returns answer with source citations

**Output**: Answers to questions

---

### Phase 3: Evaluation (Performance Assessment)

```bash
# Step 1: Generate answers (if not already done)
python3 query_rag.py --file artifacts/questions.csv
# This creates results_evaluation/questions_results.csv

# Step 2: Run evaluation
python3 src/evaluation/evaluate.py
```

**What happens**:
1. Loads questions and generated answers
2. Calculates multiple metrics
3. Generates comprehensive report
4. Optionally runs RAGAS evaluation

**Output**: Evaluation metrics and report

---

## Script Dependencies

### Dependency Graph

```
build_indices.py
    ├─→ src.parsing.pdf_parser.PDFParser
    ├─→ src.parsing.table_processor.TableProcessor
    ├─→ src.indexing.vector_store.VectorIndexBuilder
    └─→ src.indexing.facts_store.FactsStoreBuilder

query_rag.py
    └─→ src.retrieval.rag_agent.RAGAgent
        ├─→ src.retrieval.retrieval_logic.RetrievalLogic
        └─→ src.retrieval.utils

src/evaluation/evaluate.py
    ├─→ src.retrieval.rag_agent.RAGAgent
    └─→ src.evaluation.evaluate.RAGEvaluator
```

### Module Dependencies

**`src.parsing`**:
- `llama-parse`: PDF parsing
- `unstructured`: Text extraction
- `pandas`: DataFrame manipulation
- `langchain-google-vertexai`: LLM for summaries

**`src.indexing`**:
- `langchain-google-vertexai`: Embeddings and LLM
- `langchain-google-community`: BigQuery Vector Store
- `google-cloud-bigquery`: BigQuery client
- `pandas`: DataFrame manipulation
- `pandas-gbq`: DataFrame to BigQuery upload

**`src.retrieval`**:
- `langchain-google-vertexai`: LLM
- `langchain-google-community`: BigQuery Vector Store
- `google-cloud-bigquery`: BigQuery client
- `flashrank`: Optional reranking

**`src.evaluation`**:
- `pandas`: Data manipulation
- `ragas`: Optional advanced metrics

---

## Utility Scripts (Not in Main Pipeline)

### `scripts/check_gcp_connection.py`

**Purpose**: Quick sanity check for GCP/BigQuery credentials.

**Usage**:
```bash
python3 -m scripts.check_gcp_connection
```

**What it does**:
- Checks `GOOGLE_CLOUD_PROJECT` environment variable
- Tests BigQuery client connection
- Lists available datasets

**Status**: Utility script, not part of main pipeline → **Excluded from git**

---

### `scripts/test_llamaparse_tables.py`

**Purpose**: Standalone testing of LlamaParse table extraction.

**Usage**:
```bash
python3 scripts/test_llamaparse_tables.py
```

**What it does**:
- Tests LlamaParse on sample PDFs
- Extracts tables from JSON output
- Saves tables as CSV for inspection
- Provides detailed debugging information

**Status**: Testing script, not part of main pipeline → **Excluded from git**

---

## Execution Order

### First-Time Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   ```bash
   # Create .env file
   echo "GOOGLE_CLOUD_PROJECT=your-project-id" > .env
   echo "LLAMA_CLOUD_API_KEY=your-api-key" >> .env
   ```

3. **Build Indices**:
   ```bash
   python3 build_indices.py
   ```

### Regular Usage

1. **Query System**:
   ```bash
   python3 query_rag.py "Your question"
   ```

2. **Evaluate Performance** (optional):
   ```bash
   # Generate answers
   python3 query_rag.py --file artifacts/questions.csv
   
   # Evaluate
   python3 src/evaluation/evaluate.py
   ```

### Adding New PDFs

1. **Add PDFs** to `artifacts/1/` or `artifacts/2/`
2. **Run indexing**:
   ```bash
   python3 build_indices.py
   ```
   - Automatically detects new PDFs
   - Skips already-processed PDFs
   - Incrementally adds to indices

---

## File References

### Input Files

- **PDFs**: `artifacts/1/*.pdf`, `artifacts/2/*.pdf`
- **Questions**: `artifacts/questions.csv`

### Output Files

- **Processing Log**: `processing_log.json`
- **Extracted Tables**: `extracted_tables/*.csv`
- **Query Results**: `results_evaluation/questions_results.csv`
- **Evaluation Results**: `results_evaluation/evaluation_results.csv`
- **Evaluation Report**: `results_evaluation/evaluation_report.txt`

### Configuration Files

- **Environment**: `.env` (not in git)
- **Prompt Template**: `src/retrieval/prompt_template_v2.txt`
- **Dependencies**: `requirements.txt`

---

## Error Handling

### Build Indices

- **PDF Parsing Errors**: Logged, processing continues with next PDF
- **Table Processing Errors**: Skipped with warning
- **BigQuery Upload Errors**: Detailed error messages, processing continues

### Query RAG

- **Missing Indices**: Raises error (indices must be built first)
- **Query Errors**: Logged, returns error message
- **LLM Errors**: Logged, returns partial answer if possible

### Evaluation

- **Missing Results**: Prompts to generate results first
- **Metric Calculation Errors**: Logged, uses default values
- **RAGAS Errors**: Falls back to custom metrics only

---

## Performance Considerations

### Indexing Phase

- **Time per PDF**: ~30-120 seconds (depends on size and complexity)
- **Total Time**: ~1-2 hours for 40 PDFs
- **Bottlenecks**: 
  - LlamaParse API calls (rate limits)
  - Embedding generation (batched to respect limits)
  - BigQuery uploads (parallelized where possible)

### Query Phase

- **Single Query**: ~10-30 seconds
- **Bottlenecks**:
  - Semantic retrieval (vector search)
  - Table fetching (BigQuery queries)
  - LLM generation (Gemini 2.5 Pro)

### Evaluation Phase

- **Fast Mode** (with cached results): ~5-10 seconds
- **Full Mode** (re-running queries): ~1-5 minutes (depends on question count)


