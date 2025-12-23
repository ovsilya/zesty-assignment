# Parsing Module

This module is responsible for extracting structured content from PDF documents. It uses a hybrid approach combining **LlamaParse** (for structured data) and **Unstructured** (for robust text extraction).

## Overview

The parsing module extracts three types of content from PDFs:
1. **Text Elements**: Paragraphs, headings, and other text content
2. **Tables**: Structured tabular data with rows and columns
3. **Charts**: Visual charts and graphs with extracted data

## Architecture

```
PDF Files
    │
    ├─→ LlamaParse (JSON) ──→ Tables, Charts, Structured Text
    │
    └─→ Unstructured ──→ Text Elements (for DOC Index)
         │
         └─→ ParsedDocument
              ├─→ text_elements: List[ParsedElement]
              ├─→ tables: List[ParsedTable]
              └─→ charts: List[ParsedChart]
```

---

## PDF Parser (`pdf_parser.py`)

### Purpose

Extracts text, tables, and charts from PDF files using LlamaParse and Unstructured.

### Key Classes

#### `PDFParser`

Main class for parsing PDF documents.

**Initialization:**
```python
parser = PDFParser()
```

**Requirements:**
- `LLAMA_CLOUD_API_KEY` environment variable must be set
- `llama-parse` package must be installed

### Data Structures

#### `ParsedElement`

Represents a text element extracted from a PDF.

```python
@dataclass
class ParsedElement:
    type: str              # e.g., "paragraph", "heading"
    text: str              # Text content
    page_number: Optional[int]
    metadata: Dict[str, Any]  # Source, parser info, etc.
```

#### `ParsedTable`

Represents a table extracted from a PDF.

```python
@dataclass
class ParsedTable:
    rows: Optional[List[List[Any]]]  # Structured rows (preferred)
    csv: Optional[str]               # CSV string (alternative)
    page_number: Optional[int]
    metadata: Dict[str, Any]         # Source, parser, isPerfectTable, etc.
```

**Note:** Uses `rows` from LlamaParse JSON (preferred) or `csv` as fallback.

#### `ParsedChart`

Represents a chart extracted from a PDF.

```python
@dataclass
class ParsedChart:
    data: Optional[Dict[str, Any]]   # Chart data points
    chart_type: Optional[str]         # e.g., "bar", "line", "pie"
    title: Optional[str]              # Chart title
    page_number: Optional[int]
    metadata: Dict[str, Any]          # Source, parser, raw item data
```

#### `ParsedDocument`

Complete parsed representation of a PDF.

```python
@dataclass
class ParsedDocument:
    pdf_path: str
    text_elements: List[ParsedElement]
    tables: List[ParsedTable]
    charts: List[ParsedChart]
```

### Methods

#### `parse_pdf(pdf_path: Path) -> ParsedDocument`

Parses a single PDF file and returns a `ParsedDocument`.

**Process:**
1. **LlamaParse JSON Extraction**:
   - Creates LlamaParse parser with JSON output
   - Extracts structured data (tables, charts)
   - Uses async API (`aget_json()`)
   
2. **Text Extraction**:
   - Uses Unstructured (`partition_pdf`) for robust text extraction
   - High-resolution strategy with table inference
   - Skips table elements (handled by LlamaParse)

3. **Data Extraction**:
   - Extracts tables from JSON using `_extract_tables_from_json()`
   - Extracts charts from JSON using `_extract_charts_from_json()`
   - Extracts text from Unstructured using `_extract_text_with_unstructured()`

**Returns:** `ParsedDocument` with all extracted content

#### `parse_directory(pdf_dir: Path) -> List[ParsedDocument]`

Parses all PDF files in a directory.

**Process:**
- Finds all `.pdf` files in the directory
- Parses each file sequentially
- Returns list of `ParsedDocument` objects

### LlamaParse Configuration

The parser uses LlamaParse with the following settings:

```python
LlamaParse(
    api_key=api_key,
    result_type="json",              # JSON output for structured data
    verbose=False,
    num_workers=1,
    extract_charts=True,              # Enable chart extraction
    system_prompt="Convert visual bar charts to structured tables...",
    auto_mode=True,
    outlined_table_extraction=True,  # Better table detection
    adaptive_long_table=True         # Handle long tables
)
```

### Text Extraction Strategy

#### LlamaParse Text (Primary)

- Uses page-level markdown (`page["md"]` or `page["markdown"]`)
- Strips HTML/markdown table blocks (tables handled separately)
- Splits into paragraphs by blank lines
- Filters out table-like content (lines starting with `|`)

#### Unstructured Text (Fallback)

- Uses `partition_pdf` with high-resolution strategy
- Extracts text elements with categories (Title, NarrativeText, etc.)
- Skips table elements (handled by LlamaParse)
- Preserves page numbers and metadata

### Table Extraction

Tables are extracted from LlamaParse JSON using the `items` array:

**Process:**
1. Iterates through pages in JSON
2. Finds items with `type="table"`
3. Extracts `rows` (list of lists) - **preferred format**
4. Validates row structure
5. Creates `ParsedTable` with metadata

**Metadata Includes:**
- `isPerfectTable`: Boolean indicating table quality
- `parser`: "llamaparse"
- `source`: PDF path

**Note:** The parser intentionally uses only `rows` from LlamaParse JSON, ignoring `csv`, `md`, or `html` representations for simplicity and determinism.

### Chart Extraction

Charts are extracted from LlamaParse JSON:

**Process:**
1. Iterates through pages in JSON
2. Finds items with `type` in `['chart', 'figure', 'graph', 'plot', 'visualization']`
3. Extracts:
   - `data`: Chart data points
   - `chart_type`: Type of chart
   - `title`: Chart title or caption
4. Creates `ParsedChart` with metadata

**Chart Data Format:**
- Structured as dictionary when available
- Falls back to raw data if structured format unavailable

---

## Table Processor (`table_processor.py`)

### Purpose

Converts `ParsedTable` instances into cleaned pandas DataFrames and generates LLM summaries for semantic indexing.

### Key Classes

#### `TableProcessor`

Main class for processing tables.

**Initialization:**
```python
processor = TableProcessor(model_name="gemini-2.5-flash")
```

**Key Components:**
- **LLM**: Vertex AI Gemini 2.5 Flash (for summaries)
- **Summary Prompt**: Template for generating table summaries

### Data Structures

#### `ProcessedTable`

Final processed table ready for indexing.

```python
@dataclass
class ProcessedTable:
    pdf_path: str
    table_id: str                    # Unique identifier
    page_number: Optional[int]
    dataframe: pd.DataFrame         # Cleaned DataFrame
    summary: str                     # LLM-generated summary
```

### Methods

#### `process_documents(docs: List[ParsedDocument]) -> List[ProcessedTable]`

Processes all tables from a list of parsed documents.

**Process:**
1. Iterates through all documents and their tables
2. For each table:
   - Converts to DataFrame (`_to_dataframe()`)
   - Normalizes DataFrame (`_normalize()`)
   - Generates LLM summary (`_summarize()`)
   - Creates `ProcessedTable`

**Returns:** List of `ProcessedTable` objects

#### `_to_dataframe(table: ParsedTable) -> pd.DataFrame`

Converts a `ParsedTable` to a pandas DataFrame.

**Header Detection Strategy:**

1. **Key-Value Tables** (2 columns, explicitly imperfect):
   - Uses generic headers: `col1`, `col2`
   - No header row detection

2. **Standard Tables** (heuristic detection):
   - **First row as header** if:
     - First row is mostly text (≥50% has letters)
     - Second row is mostly numeric (≥50% is numeric-like)
   - **Generic headers** (`col1`, `col2`, ...) otherwise

3. **Single Row Tables**:
   - Always uses generic headers (cannot reliably detect)

**Numeric Detection:**
- Strips commas, dollar signs, percent signs
- Checks if remaining characters are digits, dots, or dashes
- Handles currency, percentages, decimals

**Example:**
```python
# Input rows:
[["Coverage A", "Base Rate"], ["$500,000", "$293.00"]]

# Detected as header + data:
# Columns: ["Coverage A", "Base Rate"]
# Data: [["$500,000", "$293.00"]]
```

#### `_normalize(df: pd.DataFrame) -> pd.DataFrame`

Basic DataFrame cleanup.

**Process:**
1. Drops completely empty rows
2. Drops completely empty columns
3. Strips whitespace from column names
4. Returns cleaned DataFrame

#### `_summarize(df: pd.DataFrame, context: str) -> str`

Generates a concise LLM summary of the table.

**Summary Prompt:**
```
You are summarizing a table extracted from an insurance filing PDF.
Provide a concise 2–3 sentence summary that includes:
1) What the table describes (subject and granularity),
2) Key columns/dimensions, and
3) Any notable categories or ranges.

Context: {context}

Table (markdown):
{table}

Summary:
```

**Context Includes:**
- Document name
- Page number

**Example Summary:**
> "This table shows Hurricane Base Rates for different policy types. It contains a single row with a Hurricane column showing the base rate value of $293. The table is from the rate pages document and represents a fixed base rate for hurricane coverage."

### Table ID Generation

Table IDs are generated using the pattern:
```
{PDF_stem}_table_{index}
```

**Example:**
- PDF: `(215004905-180407973)-CT Homeowners MAPS Rate Pages Eff 8.18.25 v3.pdf`
- Table ID: `(215004905-180407973)-CT Homeowners MAPS Rate Pages Eff 8.18.25 v3_table_0`

---

## Data Flow

### Complete Parsing Pipeline

```
1. PDF File
   │
   ├─→ LlamaParse (JSON)
   │   ├─→ Extract tables (rows)
   │   ├─→ Extract charts (data, type, title)
   │   └─→ Extract text (markdown)
   │
   └─→ Unstructured
       └─→ Extract text elements (paragraphs, headings)

2. ParsedDocument
   ├─→ text_elements: List[ParsedElement]
   ├─→ tables: List[ParsedTable]
   └─→ charts: List[ParsedChart]

3. Table Processing
   └─→ ProcessedTable
       ├─→ dataframe: pd.DataFrame (cleaned)
       └─→ summary: str (LLM-generated)
```

### Text Extraction Flow

```
PDF Page
    │
    ├─→ LlamaParse JSON
    │   └─→ page["md"] or page["markdown"]
    │       └─→ Remove table blocks
    │           └─→ Split into paragraphs
    │               └─→ ParsedElement (type="paragraph")
    │
    └─→ Unstructured partition_pdf
        └─→ Elements (Title, NarrativeText, etc.)
            └─→ Filter out tables
                └─→ ParsedElement (type=category)
```

### Table Extraction Flow

```
PDF Page
    │
    └─→ LlamaParse JSON
        └─→ page["items"] (type="table")
            └─→ Extract "rows" (list of lists)
                └─→ ParsedTable
                    │
                    └─→ TableProcessor
                        ├─→ Convert to DataFrame
                        ├─→ Detect headers
                        ├─→ Normalize
                        └─→ Generate summary
                            └─→ ProcessedTable
```

---

## Configuration

### Environment Variables

- `LLAMA_CLOUD_API_KEY`: Required for LlamaParse API access

### LlamaParse Settings

- **Result Type**: JSON (structured data)
- **Chart Extraction**: Enabled
- **Table Extraction**: Outlined tables, adaptive long tables
- **System Prompt**: Guides chart-to-table conversion

### Unstructured Settings

- **Strategy**: High-resolution (`hi_res`)
- **Table Inference**: Enabled (for detection, not extraction)
- **Image Extraction**: Disabled

---

## Usage Examples

### Parse Single PDF

```python
from src.parsing.pdf_parser import PDFParser
from pathlib import Path

parser = PDFParser()
doc = parser.parse_pdf(Path("document.pdf"))

print(f"Text elements: {len(doc.text_elements)}")
print(f"Tables: {len(doc.tables)}")
print(f"Charts: {len(doc.charts)}")
```

### Parse Directory

```python
from src.parsing.pdf_parser import PDFParser
from pathlib import Path

parser = PDFParser()
docs = parser.parse_directory(Path("artifacts/1"))

for doc in docs:
    print(f"{doc.pdf_path}: {len(doc.tables)} tables")
```

### Process Tables

```python
from src.parsing.table_processor import TableProcessor
from src.parsing.pdf_parser import PDFParser

# Parse documents
parser = PDFParser()
docs = parser.parse_directory(Path("artifacts/1"))

# Process tables
processor = TableProcessor()
processed_tables = processor.process_documents(docs)

for table in processed_tables:
    print(f"{table.table_id}: {len(table.dataframe)} rows")
    print(f"Summary: {table.summary[:100]}...")
```

### Convenience Function

```python
from src.parsing.pdf_parser import parse_all_artifacts
from pathlib import Path

# Parse all PDFs in artifacts/1 and artifacts/2
docs = parse_all_artifacts(Path("."))
```

---

## Error Handling

### LlamaParse Errors

- **API Errors**: Logged, parsing continues with available data
- **JSON Parsing Errors**: Handled with fallbacks
- **Missing Data**: Returns empty lists for missing content types

### Unstructured Errors

- **Import Errors**: Logged, text extraction skipped
- **Partition Errors**: Logged, returns empty text elements list

### Table Processing Errors

- **Empty Tables**: Skipped with warning
- **Header Detection Failures**: Falls back to generic headers
- **LLM Summary Errors**: Logged, processing continues

---

## Performance Considerations

### LlamaParse

- **API Rate Limits**: Consider LlamaParse API quotas
- **Processing Time**: ~5-30 seconds per PDF (depends on size and complexity)
- **Async Processing**: Uses async API for better performance

### Unstructured

- **Processing Time**: ~2-10 seconds per PDF
- **Memory Usage**: High-resolution strategy uses more memory
- **CPU Usage**: Can be CPU-intensive for large PDFs

### Table Processing

- **LLM Summaries**: ~1-2 seconds per table
- **DataFrame Operations**: Minimal overhead
- **Batch Processing**: Processes tables sequentially

### Optimization Tips

1. **Parallel Processing**: Parse multiple PDFs in parallel
2. **Caching**: Cache parsed documents to avoid re-parsing
3. **Selective Processing**: Process only needed content types
4. **Batch LLM Calls**: Consider batching summary generation (future improvement)

---

## Dependencies

### Required

- `llama-parse`: LlamaParse API client
- `unstructured`: PDF text extraction (optional but recommended)
- `pandas`: DataFrame manipulation
- `langchain-google-vertexai`: LLM for summaries

### Optional

- `unstructured[pdf]`: PDF partitioning support

---

## Known Limitations

1. **LlamaParse API**: Requires internet connection and API key
2. **Table Quality**: Depends on PDF quality and LlamaParse accuracy
3. **Chart Extraction**: Limited to charts that LlamaParse can detect
4. **Text Extraction**: May miss some text in complex layouts
5. **Header Detection**: Heuristic-based, may not work for all table formats

---

## Future Improvements

1. **Better Header Detection**: Use LLM for header detection
2. **Table Merging**: Detect and merge tables split across pages
3. **Chart Data Extraction**: Better extraction of chart data points
4. **Parallel Processing**: Parse multiple PDFs concurrently
5. **Caching**: Cache parsed results to avoid re-parsing
6. **Error Recovery**: Better handling of corrupted or problematic PDFs
7. **Progress Tracking**: Better progress reporting for large batches

