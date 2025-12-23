## Engineering the Indices – Design & Thought Process

### Overview

This document summarizes my thought process while designing and implementing the indexing pipeline for ~40 regulatory PDFs.  
It focuses on how the architecture evolved, why certain choices were made over alternatives, and the non-trivial issues I had to solve around PDF parsing and indexing (omitting routine “fix this error” debugging).

---

## Initial Design Exploration: Three Indexing Solutions

### High‑Level Goal

The assignment asked for an **indexing layer** capable of:

- Handling **unstructured regulatory PDFs** that contain dense text, complex tables, and charts.
- Producing:
  - A **facts store** of structured tabular data.
  - A **vector index** over text, tables, and chart summaries for semantic lookup.
- Staying within realistic **cloud, cost, and ops constraints**.

From that, I explored three indexing architectures:

### Solution 1 – BigQuery‑Centric Indexing (Chosen)

- **Facts Store**: BigQuery native tables per logical table (rate tables, disruption tables, etc.).
- **Vector Store**: BigQuery as a vector database via LangChain’s `BigQueryVectorStore` + Vertex AI embeddings.
- **Characteristics**:
  - Single storage backend (BigQuery) for **both** structured facts and vector embeddings.
  - Simple operational story: no separate vector DB, leverage existing GCP infra.
  - Direct SQL access to both raw facts and semantic index (for debugging and analytics).

**Why I chose this**:
- It aligns with the assignment’s GCP context.
- Minimizes moving parts while still being realistic for production.
- BigQueryVectorStore + Vertex AI gives a well‑supported, managed path with reasonable performance for this scale.

### Solution 2 – Vertex AI Matching Engine + BigQuery

- **Facts Store**: BigQuery.
- **Vector Store**: Vertex AI Matching Engine (managed ANN index), with embeddings stored separately.
- **Pros**:
  - Strong retrieval performance at scale; production‑grade vector search.
- **Cons**:
  - More infra to configure (index endpoints, deployment).
  - Overkill for the current data volume; less “self‑contained” than Solution 1.

**Why I didn’t pick it for this stage**:
- Higher setup and operational complexity for limited benefit at this scale.
- BigQueryVectorStore already meets the requirements while keeping everything in one place.

### Solution 3 – External OSS Vector DB (e.g., pgvector/Chroma) + BigQuery

- **Facts Store**: BigQuery.
- **Vector Store**: Open‑source vector DB (self‑hosted or managed).
- **Pros**:
  - Flexibility and portability; can run locally.
- **Cons**:
  - More infra to manage; not as aligned with a GCP‑native architecture.
  - Fragmented data plane (vectors outside BigQuery, facts inside).

**Why I deprioritized it**:
- Less aligned with a cloud‑native GCP stack.
- Adds infra complexity without clear benefits compared to BigQueryVectorStore at this scale.

**Conclusion**: I proceeded with **Solution 1 (BigQuery‑centric)** and designed everything around **BigQuery facts tables + BigQueryVectorStore**.

---

## Evolution of PDF Parsing Strategy

A core challenge was extracting **reliable structured data** and **high‑quality text chunks** from messy regulatory PDFs that mix prose, tables, and charts.

### Phase 1 – Baseline: Unstructured + LlamaParse (Tables Only)

**Initial approach**:

- **Text**: Use the `unstructured` library in high‑resolution mode for plain text extraction.
- **Tables**: Use LlamaParse (initially `result_type="markdown"`) to pull out tables.

Rationale:

- `unstructured` was a known quantity for text extraction from PDFs.
- LlamaParse was clearly strong for **table detection** and **structure**.

Issues & limitations:

- Markdown output from LlamaParse was harder to parse robustly into DataFrames.
- Having **two different parsing stacks** (Unstructured + LlamaParse) made it harder to reason about end‑to‑end consistency.

This led me to explore a more unified parsing strategy.

### Phase 2 – Attempted Unification: LlamaParse JSON for Everything

I then tried to **standardize on LlamaParse JSON**:

- Switched LlamaParse to `result_type="json"`.
- Implemented `_extract_text_from_json` that:
  - Traversed page‑level `items` (`text`, `heading`, `paragraph`, `list_item`, etc.).
  - Also experimented with page‑level `text` and `md` (markdown) fields.

Goal:

- **One parser**, one JSON schema, single source of truth.
- Use LlamaParse JSON for:
  - Text.
  - Tables.
  - (Later) charts.

What I found:

- For these regulatory PDFs, LlamaParse’s page‑level `md` / `text` sometimes contained **LLM‑like summaries** or **HTML tables embedded in the text**.
- Pulling text from those fields without very careful filtering led to:
  - Duplicated table content in text.
  - Less faithful raw text than Unstructured produced.

I tried several variants:

- Using only page‑level `text`.
- Using only page‑level `md` and stripping HTML/markdown tables.
- Combining `items` + page‑level text with heuristics.

Despite these attempts, **Unstructured still produced the most reliable “plain reading text”** for the use case.

### Phase 3 – Final Parsing Split: Unstructured for Text, LlamaParse for Tables/Charts

Given the above, I settled on a **hybrid but clear separation**:

- **Text extraction**:
  - Use `unstructured` in hi‑res mode.
  - Convert each text block into a `ParsedElement` with page/position metadata.
- **Structured extraction (tables & charts)**:
  - Use LlamaParse with `result_type="json"`.
  - Configure it specifically for structured outputs:
    - `extract_charts=True`
    - `outlined_table_extraction=True`
    - `adaptive_long_table=True`
    - A custom `system_prompt` to encourage chart‑to‑table conversion:
      - “Convert visual bar charts to structured tables with categories/bins as rows and values as columns. Check for the axis and legends.”

This gave me:

- High‑quality raw text for semantic indexing.
- High‑fidelity, machine‑readable tables (and chart‑derived tables) for the facts store.

---

## Table & Chart Extraction Design

### Parsing Tables from LlamaParse JSON

Key decisions for table extraction:

- **Source of truth**: Use only items where `type == "table"` in the JSON.
- **Primary structure field**: Use `rows` (list of lists) as the canonical representation.
- **Chart tables**:
  - For bar charts, LlamaParse (with the `system_prompt`) tends to output an equivalent `type: "table"` with `rows` representing the chart data.
  - I treat these identically to other tables; they are just another `ParsedTable`.

I experimented briefly with using the `csv` field as a fallback when `rows` looked corrupted in some specific cases, but ultimately reverted to a **simpler and more predictable rule**:

- **Use `rows` only** for the pipeline.
- Accept that occasional upstream oddities are better handled by strictness and logging than by over‑engineering the extraction logic.

### Header Handling Strategy

Tables in these PDFs are not uniform:

- Some have clear column headers (e.g., “Rate Change Range”, “Count”).
- Some are **key‑value** style, often two columns.
- Others appear without explicit headers.

I wanted:

- To **preserve real headers** wherever they exist.
- Use generic headers only when the table clearly has none.

Final rules in `TableProcessor._to_dataframe` (and mirrored in the CSV export helper):

1. **Guardrail**: If `rows` is empty or malformed → return empty DataFrame.
2. **Key‑value tables**:
   - If `isPerfectTable` is explicitly `False` **and** there are exactly 2 columns:
     - Treat as a key‑value table.
     - Assign generic headers: `col1`, `col2`.
3. **Heuristic header detection (all other tables)**:
   - Normalize all cells to strings.
   - Look at the first two rows:
     - First row is header‑like if **most cells contain letters**.
     - Second row is data‑like if **most cells are numeric-ish** (numbers, %, currency).
   - If both conditions hold:
     - Use row 1 as headers; backfill empty header cells as `col{i}`.
     - Data starts from row 2.
   - Otherwise:
     - No strong evidence of a header row → use generic `col1..colN` for all columns.

4. **Single‑row tables**:
   - Can’t reliably detect a header; fall back to generic `col1..colN`.

This ensures:

- Tables such as:

  - `"Rate Change Range", "Count", ...`
  - `"Percent‑Change Range", "Number of Insureds in Range"`

  retain their original headers.

- Truly headerless or key‑value tables get generic `colX` columns, but **only when needed**.

### Chart Handling

LlamaParse with `extract_charts=True` and the custom `system_prompt` gives me:

- For many bar charts, **a companion table** describing:
  - Category/bucket.
  - Count/value.

I handle charts at two levels:

- **Structured level**:
  - Some charts appear directly as `type: "table"` items with `rows` and `csv`.
  - These go through the same `ParsedTable → DataFrame` flow as regular tables.
- **Semantic level**:
  - I wrap chart metadata into `ParsedChart` objects and then into LangChain `Document`s with readable text:
    - Title, chart type, and a compact textual summary of `data` (up to a few key entries).
  - These are added to the **vector index** so that chart summaries are searchable.

---

## Text Chunking & Embedding Strategy

### Text Chunking

Once I had reliable text elements from Unstructured, I needed them in a form suitable for embeddings and semantic search.

I used LangChain’s `RecursiveCharacterTextSplitter` with:

- Chunk size ≈ 1000 characters.
- Overlap ≈ 100 characters.
- Separators: paragraph breaks, line breaks, sentence breaks, and spaces.

Rationale:

- 1000 characters is a good compromise between:
  - Enough context to answer questions about a section or table description.
  - Not so large as to blow up token limits or dilute retrieval.

Each chunk becomes a `Document` with metadata:

- `source`: PDF path.
- `page_number`: from the original text block.
- `element_type`: text/heading/etc.
- `chunk_id`: index within the page/element.

### Embeddings & Token Limits

Embeddings are generated via `VertexAIEmbeddings` (`text-embedding-005`) and stored in BigQuery using `BigQueryVectorStore`.

I hit two main constraints:

- **Max instances per prediction** (~250).
- **Max tokens per request** (~20k).

To respect both, I implemented a **batching helper**:

- `_add_documents_batched(store, docs, max_instances=200, max_chars=60000)`:
  - Builds batches of `Document`s such that:
    - Each batch has ≤ 200 instances.
    - The **sum of character counts** per batch is bounded (used as rough proxy for tokens).
  - Very long documents are sent alone.
- Both the **incremental add** (`add_document_to_index`) and **batch build** (`build_index`) use this helper.

This eliminated `INVALID_ARGUMENT` errors due to token limits and made the embedding step robust across longer PDFs.

---

## BigQuery Facts Store Design

For the **facts store**, I focused on:

- **Readable, normalized tables** in BigQuery.
- A path from `ParsedTable` → `DataFrame` → BigQuery that:
  - Preserves numeric types where possible.
  - Has safe, sanitized column names.
  - Is idempotent for re‑runs.

Key aspects:

- **Column sanitization**:
  - Strip problematic characters.
  - Flatten nested/multi‑index columns if needed.
  - Ensure unique column names (e.g., suffix duplicates).
- **Write semantics**:
  - Drop any existing table before re‑loading to avoid schema conflicts.
- **Types**:
  - Infer from DataFrame dtypes, but fail gracefully if specific cells cause trouble (e.g., mixed types).

Additionally, I added a utility to **save raw extracted tables as CSVs** before any normalization:

- Mirrors the header detection logic from `TableProcessor`.
- Outputs CSVs to `extracted_tables/` per PDF, per table.
- This served as a **debugging and auditing tool**:
  - Compare raw LlamaParse output vs. what ends up in BigQuery.

---

## Vector Index in BigQuery

The **DOC Index** is stored in BigQuery through `BigQueryVectorStore` and receives three types of content:

1. **Text chunks** (from Unstructured):
   - Large majority of the index.
2. **Table summaries**:
   - Each processed table is summarized by an LLM (Gemini) into a concise text.
   - These summaries become `Document`s with metadata including `table_id` and `page_number`.
   - Helps retrieval when the user’s query is about the semantics of a table, not its raw cells.
3. **Chart descriptions**:
   - Generated from `ParsedChart` objects by turning chart metadata and data into a compact, human‑readable description.
   - Indexed as `element_type="chart"`.

For each PDF in the pipeline:

- A single call to `add_document_to_index(parsed_doc, processed_tables, store)`:
  - Generates text docs, table summary docs, and chart docs.
  - Uses the batched embed/upload helper to push them into BigQuery.

This yields a **unified semantic index** spanning text, table summaries, and chart descriptions, all in one BigQuery table.

---

## Orchestration, Logging, and Robustness

### Per‑PDF End‑to‑End Processing

Instead of processing all PDFs in one global pass per stage, I refactored the orchestration to:

- **Process one PDF at a time, end‑to‑end**:

  1. Parse PDF (text, tables, charts).
  2. Save raw tables as CSV (for inspection).
  3. Convert tables → DataFrames → BigQuery.
  4. Generate embeddings and update the vector index.

Benefits:

- Limits scope of failures to a single PDF.
- Makes re‑runs cheaper and easier to debug (you can target a single problematic PDF).

### Processing Log

To avoid wasting LlamaParse free tier and to make the pipeline resumable:

- Introduced a `processing_log.json` file with entries per PDF:
  - File name, timestamp, status, any summary statistics (e.g., number of tables).
- On each run:
  - Skip PDFs already marked as successfully processed, unless an explicit override is set (e.g., `FORCE_REPROCESS` env var).

This ensures that:

- Re‑running the pipeline is fast and idempotent.
- New or updated PDFs can be processed incrementally.

---

## Key Trade‑Offs & Rationale

- **BigQuery as both facts and vector store**:
  - Chosen over more complex or external vector DBs to keep the system **simple, GCP‑native, and easy to operate**, while still realistic for production.
- **Unstructured for text vs. LlamaParse JSON for text**:
  - Multiple experiments with LlamaParse JSON for text showed quality and duplication issues for this specific corpus.
  - I prioritized **faithfulness of text** and **predictability**, hence kept Unstructured for the text layer and reserved LlamaParse for structured content.
- **Using only `rows` for tables, with clear header heuristics**:
  - Overly clever fallbacks (e.g., mixing `csv`, `html`, or attempting to detect chart visualizations heuristically) made the pipeline brittle and hard to reason about.
  - Settling on `rows` with a transparent header heuristic gives a **clean mental model** and tables that are easy to inspect and debug.
- **Per‑PDF end‑to‑end processing**:
  - Slightly more orchestration code, but significantly simpler failure modes and resource usage, which is important when parsing heavier PDFs and dealing with API limits.
- **Batching for embeddings**:
  - Necessary to comply with Vertex AI limits.
  - The chosen batching uses both instance count and approximate token budget, making the system robust even as individual documents or chunks size vary.

---

## Representative Queries / Prompts Used During Development

The employer requested the queries used during development. Below are representative prompts I used with an LLM while engineering the indexing pipeline (edited slightly for clarity, but structurally faithful to what I asked):

- **LlamaParse configuration and text vs. JSON**  
  - “Given that I’m using LlamaParse, when should I use `result_type="markdown"` vs `result_type="json"` if I need both structured tables and raw text? How do I reliably extract text portions from the JSON output without duplicating tables?”
- **Unstructured vs. LlamaParse for text**  
  - “I’m currently using Unstructured for text and LlamaParse for tables. What are the trade‑offs of unifying everything on LlamaParse JSON vs keeping this hybrid approach? For regulatory PDFs with dense tables, what would you recommend?”
- **Table extraction and headers**  
  - “Here is a sample of LlamaParse JSON with multiple `type:"table"` items (pasted). Design a robust `_to_dataframe` function that prefers `rows`, preserves genuine headers, and falls back to `col1`, `col2`, etc., only when there are no clear column names.”
- **Chart‑to‑table conversion**  
  - “Some PDFs have bar charts. I want LlamaParse to extract these as structured tables. How should I configure `extract_charts` and `system_prompt` so that charts are converted into ‘categories vs values’ tables that I can load into pandas?”
- **Handling LlamaParse mixed outputs (text + HTML tables)**  
  - “In LlamaParse JSON, page‑level `md` and `text` fields often include `<table>` HTML that duplicates structured table items. How can I strip or ignore these tables while still keeping the surrounding prose for text indexing?”
- **BigQueryVectorStore setup**  
  - “I’m using LangChain’s `BigQueryVectorStore` with `VertexAIEmbeddings`. Show an example of constructing the store and adding `Document`s with metadata like page number and table ID.”
- **Embedding limits and batching**  
  - “I’m seeing `400 INVALID_ARGUMENT: 250 instances allowed per prediction, actual 1087`. How can I batch `store.add_documents` calls to stay under both the instance and token limits for Vertex AI embeddings?”
- **Pipeline orchestration and processing log**  
  - “I want to process ~40 PDFs end‑to‑end one by one and avoid re‑processing the same file twice. Propose a simple logging mechanism (e.g., JSON log) to track processed PDFs and an optional flag to force re‑processing for debugging.”
- **CSV export for debugging tables**  
  - “Given `ParsedTable` objects with `rows` and metadata, write a helper that saves each table to a CSV file with sane headers, mirroring the same header detection logic used for DataFrames, so I can visually inspect the raw extractions.”

These queries guided the design of:

- The **hybrid parsing strategy** (Unstructured + LlamaParse).
- The **table and chart handling**.
- The **BigQuery facts store** and **vector index**.
- The **orchestration and resilience** of the pipeline.

---

This concludes the summary of how I engineered the indexing pipeline and how the design evolved from initial architectural options to the final, robust PDF processing flow used to index all 40 PDFs.