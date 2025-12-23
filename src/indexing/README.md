# Indexing Module

This module is responsible for building and maintaining the two core indices used by the RAG system:

1. **DOC Index** (`vector_store.py`): Semantic vector store for text chunks, table summaries, and chart descriptions
2. **FACTS Store** (`facts_store.py`): Structured BigQuery tables for precise quantitative queries

## Overview

The indexing module transforms parsed PDF content into searchable indices:
- **Text content** → Vector embeddings → DOC Index (semantic search)
- **Table data** → BigQuery tables → FACTS Store (structured queries)
- **Table summaries** → Vector embeddings → DOC Index (semantic search for table discovery)

## Architecture

```
Parsed Documents (from parsing/)
    │
    ├─→ Text Elements ──→ Vector Embeddings ──→ DOC Index (BigQuery Vector Store)
    │
    ├─→ Table Summaries ──→ Vector Embeddings ──→ DOC Index (BigQuery Vector Store)
    │
    └─→ Table DataFrames ──→ BigQuery Tables ──→ FACTS Store
```

---

## Vector Store (`vector_store.py`)

### Purpose

Builds the **DOC Index** - a semantic vector store that enables similarity search over text content, table summaries, and chart descriptions.

### Key Classes

#### `VectorIndexBuilder`

Main class for building and maintaining the vector index.

**Initialization:**
```python
builder = VectorIndexBuilder(
    project_id="your-project-id",
    dataset_name="rag_dataset",
    location="US"
)
```

**Key Components:**
- **Text Splitter**: `RecursiveCharacterTextSplitter` with chunk size 1000, overlap 100
- **Embeddings Model**: Vertex AI `text-embedding-005`
- **Vector Store**: BigQuery Vector Store (LangChain-compatible)

### Methods

#### `build_index(parsed_docs, processed_tables, table_name="doc_index")`

Creates or updates the DOC Index in batch mode.

**Process:**
1. Extracts text chunks from parsed documents
2. Creates documents from table summaries
3. Generates embeddings using Vertex AI
4. Uploads to BigQuery Vector Store in batches

**Batch Processing:**
- Respects Vertex AI limits:
  - Max ~250 instances per prediction
  - Max ~20k tokens per request (approximated via character count)
- Batches documents to stay within limits

**Returns:** BigQuery Vector Store instance

#### `add_document_to_index(parsed_doc, processed_tables, store)`

Adds a single document's content to the vector index incrementally.

**Use Case:** Incremental indexing when processing new PDFs

**Process:**
1. Extracts text chunks from the document
2. Extracts table summaries from the document's tables
3. Extracts chart summaries from the document's charts
4. Batches and adds to the vector store

#### `_add_documents_batched(store, docs, max_instances=200, max_chars=60000)`

Internal method that batches documents to respect Vertex AI API limits.

**Batching Strategy:**
- Groups documents until either:
  - Instance count exceeds `max_instances` (default: 200)
  - Character count exceeds `max_chars` (default: 60,000)
- Sends batches sequentially to avoid API errors

### Document Structure

Documents stored in the vector index have the following structure:

**Text Chunks:**
```python
Document(
    page_content="...text content...",
    metadata={
        "source": "path/to/document.pdf",
        "page_number": 5,
        "element_type": "paragraph",
        "chunk_id": 0
    }
)
```

**Table Summaries:**
```python
Document(
    page_content="...LLM-generated summary...",
    metadata={
        "source": "path/to/document.pdf",
        "page_number": 10,
        "element_type": "table_summary",
        "table_id": "document_table_3"
    }
)
```

**Chart Descriptions:**
```python
Document(
    page_content="Chart title: ... | Chart type: bar | Chart data: ...",
    metadata={
        "source": "path/to/document.pdf",
        "page_number": 15,
        "element_type": "chart",
        "chart_type": "bar"
    }
)
```

### Text Splitting Strategy

- **Chunk Size**: 1000 characters
- **Overlap**: 100 characters
- **Separators**: `["\n\n", "\n", ". ", " ", ""]` (in order of preference)

This ensures:
- Natural boundaries (paragraphs, sentences)
- Context preservation through overlap
- Efficient embedding generation

### Embeddings

- **Model**: Vertex AI `text-embedding-005`
- **Dimensions**: 768 (model default)
- **Storage**: BigQuery Vector Store with automatic vector similarity search

---

## Facts Store (`facts_store.py`)

### Purpose

Builds the **FACTS Store** - structured BigQuery tables that store full table data for precise SQL-like queries and quantitative analysis.

### Key Classes

#### `FactsStoreBuilder`

Main class for uploading processed tables to BigQuery.

**Initialization:**
```python
builder = FactsStoreBuilder(
    project_id="your-project-id",
    dataset_name="rag_dataset"
)
```

**Key Components:**
- **BigQuery Client**: Google Cloud BigQuery client
- **Dataset Management**: Automatically creates dataset if it doesn't exist

### Methods

#### `store_table(table) -> str`

Stores a single processed table as a BigQuery table.

**Process:**
1. **Sanitizes table ID**: Converts to BigQuery-compliant table name
   - Only letters, numbers, underscores
   - Must start with letter or underscore
   - Max 1024 characters (limited to 200 for readability)
   - Prefix: `table_`
2. **Adds metadata columns**:
   - `meta_source_pdf`: Source PDF path
   - `meta_table_id`: Original table ID
   - `meta_page_number`: Page number where table was found
3. **Sanitizes column names**: Ensures BigQuery compliance
4. **Flattens DataFrame**: Handles MultiIndex columns, nested structures
5. **Validates data types**: Ensures BigQuery-compatible types
6. **Uploads to BigQuery**: Creates or overwrites table

**Returns:** Full BigQuery table ID (e.g., `project.dataset.table_name`)

#### `store_all(tables) -> List[Dict[str, str]]`

Stores multiple tables in batch.

**Returns:** List of dictionaries with:
- `table_id`: Original table ID
- `bigquery_table`: Full BigQuery table ID
- `pdf_path`: Source PDF path

### Column Name Sanitization

BigQuery has strict requirements for column names. The `_sanitize_column_name()` method:

1. **Handles MultiIndex columns**: Flattens tuple column names with underscores
2. **Removes invalid characters**: Only allows letters, numbers, underscores
3. **Ensures valid start**: Must start with letter or underscore (not number)
4. **Avoids reserved prefixes**: 
   - `_PARTITION`, `_TABLE_`, `_FILE_`, `_ROW_TIMESTAMP`, `__ROOT__`, `_COLIDENTIFIER`
   - Adds `meta_` prefix if needed
5. **Limits length**: Max 300 characters
6. **Handles duplicates**: Appends numbers to ensure uniqueness

**Example:**
```python
"Coverage A Limit" → "Coverage_A_Limit"
"$ Amount" → "Amount"
"2023 Data" → "col_2023_Data"
"__ROOT__value" → "meta___ROOT__value"
```

### DataFrame Flattening

The `_flatten_dataframe()` method handles complex DataFrame structures:

1. **Flattens MultiIndex columns**: Converts hierarchical column names to single-level
2. **Removes duplicate columns**: Appends numbers to duplicate names
3. **Handles non-scalar values**: Converts lists, dicts, DataFrames to strings
4. **Validates structure**: Ensures all values are scalar (BigQuery requirement)

### Data Type Handling

- **Object columns**: Converted to string if they contain non-scalar values
- **Numeric types**: Preserved (int, float)
- **String types**: Preserved
- **Complex types**: Converted to string representation

### Table Naming Convention

BigQuery tables are named using the pattern:
```
table_{sanitized_table_id}
```

Where `sanitized_table_id` is derived from the original table ID:
- Original: `(215004905-180407973)-CT Homeowners MAPS Rate Pages Eff 8.18.25 v3_table_5`
- Sanitized: `table__215004905_180407973__CT_Homeowners_MAPS_Rate_Pages_Eff_8_18_25_v3_table_5`

### Metadata Columns

Every table includes three metadata columns:
- `meta_source_pdf`: Full path to source PDF (STRING)
- `meta_table_id`: Original table identifier (STRING)
- `meta_page_number`: Page number where table was found (INTEGER, nullable)

These enable:
- Source attribution in answers
- Page-level retrieval
- Table identification and tracking

---

## Data Flow

### Complete Indexing Pipeline

```
1. PDF Parsing (parsing/)
   └─→ ParsedDocument
       ├─→ text_elements: List[ParsedElement]
       ├─→ tables: List[ParsedTable]
       └─→ charts: List[ParsedChart]

2. Table Processing (parsing/)
   └─→ ProcessedTable
       ├─→ dataframe: pd.DataFrame
       └─→ summary: str (LLM-generated)

3. Indexing (indexing/)
   ├─→ VectorIndexBuilder
   │   ├─→ Text chunks → DOC Index
   │   ├─→ Table summaries → DOC Index
   │   └─→ Chart descriptions → DOC Index
   │
   └─→ FactsStoreBuilder
       └─→ Table DataFrames → FACTS Store (BigQuery tables)
```

### Incremental Indexing

The system supports incremental indexing:
- `add_document_to_index()`: Adds a single document's content
- `store_table()`: Stores a single table
- Useful for processing new PDFs without rebuilding entire index

---

## Configuration

### Environment Variables

- `GOOGLE_CLOUD_PROJECT`: Google Cloud project ID
- `LLAMA_CLOUD_API_KEY`: Required for PDF parsing (not directly used in indexing)

### BigQuery Setup

- **Dataset**: Created automatically if it doesn't exist
- **Location**: US (default, configurable)
- **Permissions**: Service account needs:
  - BigQuery Data Editor
  - BigQuery Job User
  - BigQuery User

### Vertex AI Setup

- **Embeddings API**: Must be enabled
- **Model**: `text-embedding-005` (default)
- **Quota**: Consider rate limits for large batches

---

## Usage Examples

### Building Complete Index

```python
from src.indexing.vector_store import VectorIndexBuilder
from src.indexing.facts_store import FactsStoreBuilder
from src.parsing.pdf_parser import PDFParser
from src.parsing.table_processor import TableProcessor

# Parse PDFs
parser = PDFParser()
parsed_docs = parser.parse_directory(Path("artifacts/1"))

# Process tables
processor = TableProcessor()
processed_tables = processor.process_documents(parsed_docs)

# Build indices
vector_builder = VectorIndexBuilder(project_id="your-project")
vector_store = vector_builder.build_index(parsed_docs, processed_tables)

facts_builder = FactsStoreBuilder(project_id="your-project")
stored_tables = facts_builder.store_all(processed_tables)
```

### Incremental Indexing

```python
# Add a new document
new_doc = parser.parse_pdf(Path("new_document.pdf"))
new_tables = processor.process_documents([new_doc])

# Add to vector index
vector_store = vector_builder.get_or_create_store()
vector_builder.add_document_to_index(new_doc, new_tables, vector_store)

# Add to facts store
for table in new_tables:
    facts_builder.store_table(table)
```

---

## Performance Considerations

### Vector Store

- **Batch Size**: 200 documents per batch (configurable)
- **Character Limit**: 60,000 characters per batch
- **Embedding Time**: ~1-2 seconds per batch
- **Total Time**: Depends on document count and size

### Facts Store

- **Table Upload**: ~1-5 seconds per table (depends on size)
- **Column Sanitization**: Minimal overhead
- **Data Type Conversion**: Handled automatically by pandas-gbq

### Optimization Tips

1. **Parallel Processing**: Process multiple PDFs in parallel (parsing stage)
2. **Batch Uploads**: Use `store_all()` for multiple tables
3. **Incremental Updates**: Use incremental methods for new documents
4. **Caching**: Cache parsed documents to avoid re-parsing

---

## Error Handling

### Vector Store Errors

- **API Limits**: Automatically handled by batching
- **Embedding Failures**: Logged, processing continues
- **BigQuery Errors**: Raised as exceptions

### Facts Store Errors

- **Invalid Column Names**: Automatically sanitized
- **Empty DataFrames**: Skipped with warning
- **BigQuery Upload Failures**: Detailed error messages with DataFrame info
- **Type Conversion Errors**: Problematic columns dropped with warning

---

## Dependencies

### Required

- `langchain-google-vertexai`: Vertex AI embeddings and LLM
- `langchain-google-community`: BigQuery Vector Store
- `google-cloud-bigquery`: BigQuery client
- `pandas`: DataFrame manipulation
- `pandas-gbq`: DataFrame to BigQuery upload

### Optional

- `langchain-text-splitters`: Text chunking (included in langchain)

---

## Future Improvements

1. **Parallel Embedding**: Generate embeddings in parallel for faster indexing
2. **Incremental Updates**: Better support for updating existing indices
3. **Compression**: Compress large text chunks before embedding
4. **Caching**: Cache embeddings for duplicate content
5. **Monitoring**: Add metrics for indexing performance and success rates

