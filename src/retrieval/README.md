# Retrieval Module

This module implements the core retrieval and answer generation logic for the RAG system. It provides a hybrid retrieval strategy that combines semantic search (vector similarity) with structured data retrieval (BigQuery tables) to answer complex questions requiring both semantic understanding and precise quantitative data.

## Overview

The retrieval module consists of three main components:

1. **`rag_agent.py`** - Main RAG agent orchestrator that coordinates retrieval and answer generation
2. **`retrieval_logic.py`** - Core retrieval algorithms for semantic and structured data retrieval
3. **`utils.py`** - Utility functions for value extraction, table name conversion, and helper operations

## Architecture

### RAGAgent

The `RAGAgent` class is the main entry point for querying the RAG system. It orchestrates the entire retrieval and answer generation pipeline:

- **Initialization**: Sets up vector store (BigQuery), embeddings (Vertex AI), LLM (Gemini 2.5 Pro), and retrieval logic
- **Query Processing**: Always uses hybrid approach (semantic + quantitative retrieval)
- **Answer Generation**: Synthesizes answers from retrieved context using LLM with source attribution

### RetrievalLogic

The `RetrievalLogic` class encapsulates all retrieval algorithms, making them testable and modifiable independently:

- **Semantic Retrieval**: Vector similarity search with optional reranking (FlashRank)
- **Table Discovery**: Multi-tier approach to find relevant tables
- **Table Retrieval**: Full table fetching with split table handling
- **Page/Document Expansion**: Expands retrieval to include all content from same pages/documents

### Utils

Utility functions for common operations:

- Value extraction from queries (ZIP codes, territories, percentages, model names)
- Table name conversion (metadata format ↔ BigQuery format)
- Split table detection
- Geographic term extraction

## Hybrid Retrieval Strategy

The system **always uses a hybrid approach**, combining:

1. **Semantic Retrieval** (from DOC Index)
   - Vector similarity search using embeddings
   - Optional reranking with FlashRank
   - Retrieves text chunks, table summaries, and chart descriptions

2. **Structured Data Retrieval** (from FACTS Store)
   - Value-based table search (extract specific values from query)
   - Column-based table search (find tables with relevant columns)
   - Full table retrieval (fetch entire tables, not SQL queries)

3. **Context Expansion**
   - Page-level expansion: When a chunk is retrieved, include all content from the same page
   - Document-level expansion: Include related content from the same document
   - Split table merging: Detect and merge tables split across multiple pages

## Key Features

### 1. Universal Query Handling

The system handles three types of queries without hardcoding:

- **List Queries** (e.g., "List all rating plan rules")
  - Retrieves more candidates (k=40 vs k=25)
  - Performs additional targeted searches for completeness
  - Extracts key terms from initial results for follow-up searches

- **Comparison Queries** (e.g., "Does X have higher rate than Y?")
  - Geographic boundary detection
  - Territory definition matching
  - Value-based table search for specific locations/models

- **Calculation Queries** (e.g., "Calculate premium using Base Rate and Factor")
  - Dynamic base rate table discovery
  - Dynamic factor table discovery
  - Prioritizes rate pages tables and specific document patterns

### 2. Multi-Tier Table Discovery

When searching for relevant tables, the system uses a prioritized approach:

**Priority 1: Semantic Results**
- Extract table IDs from semantic retrieval results
- These are tables whose summaries matched the query

**Priority 2: Value-Based Search**
- Extract specific values from query (ZIP codes, territories, percentages, model names)
- Search table content for these values
- Prioritizes tables with exact matches in names/columns

**Priority 3: Column-Based Search**
- Search for tables with columns matching query keywords
- Useful for finding factor tables, rate tables, etc.

**Priority 3.5: Calculation-Specific Discovery** (for calculation queries)
- Dynamically finds base rate tables based on:
  - Column names ("Hurricane", "Base Rate", "Rate")
  - Page numbers (early pages of rate pages PDFs)
  - Content patterns (numeric values in typical base rate range)
- Dynamically finds factor tables based on:
  - Column names matching factor type
  - Policy type and coverage amount patterns

**Priority 4: Keyword Matching**
- Fallback search using table names and column names
- Domain-specific keyword matching

### 3. Split Table Handling

Long tables split across multiple pages are automatically detected and merged:

- **Detection**: Tables from the same PDF with similar schemas and consecutive table numbers
- **Merging**: Combines split tables into a single logical table for the LLM
- **Metadata Preservation**: Maintains source information (page numbers, document names)

### 4. Performance Optimizations

- **Parallel Processing**: Uses `ThreadPoolExecutor` for concurrent table fetching
- **Batch Schema Fetching**: Uses `INFORMATION_SCHEMA` for faster schema retrieval
- **Caching**: Caches table list for 60 seconds to reduce API calls
- **Batched Queries**: Combines multiple metadata queries using `UNION ALL`

### 5. Page/Document-Level Expansion

When retrieving content, the system ensures comprehensive context:

- **Page Expansion**: If a text chunk or table is retrieved, all content from the same page is included
- **Document Expansion**: Related content from the same document is included
- **Value-Based Expansion**: Pages from value-based table matches trigger expansion

## How It Works

### Step-by-Step Retrieval Flow

1. **Semantic Retrieval**
   ```
   Query → Vector Search → Top K Candidates → Reranking (optional) → Top Results
   ```

2. **Table Discovery**
   ```
   Semantic Results → Extract Table IDs
   Query → Extract Values → Value-Based Search
   Query → Extract Keywords → Column-Based Search
   Query → Calculation Detection → Dynamic Base Rate/Factor Discovery
   ```

3. **Table Retrieval**
   ```
   Discovered Tables → Group Split Tables → Fetch Full Tables (parallel) → Format as Markdown
   ```

4. **Context Assembly**
   ```
   Semantic Chunks + Full Tables + Page/Document Expansion → Combined Context
   ```

5. **Answer Generation**
   ```
   Combined Context + Query → LLM Prompt → Generated Answer with Sources
   ```

### Example: Answering a Query

For the query: *"For a policyholder in ZIP Code 89110, does the resident living West of Nellis Boulevard face a higher Comprehensive rate change than East of Nellis Boulevard?"*

1. **Semantic Retrieval**: Finds text chunks about territory definitions and rate changes
2. **Value Extraction**: Extracts "89110" (ZIP code) and "Nellis Boulevard" (geographic term)
3. **Value-Based Search**: Searches tables for "89110" and "Nellis Boulevard"
4. **Geographic Search**: Performs targeted search for "West of Nellis Boulevard" and territory definitions
5. **Table Retrieval**: Fetches territory definition tables and rate change tables
6. **Page Expansion**: Includes all text chunks from pages where relevant tables were found
7. **Answer Generation**: LLM synthesizes answer using all retrieved context

## Configuration

### LLM Settings

- **Model**: Gemini 2.5 Pro
- **Max Output Tokens**: 8192 (prevents truncation)
- **Temperature**: 0.1 (for consistent, factual responses)
- **Prompt Template**: Loaded from `prompt_template_v2.txt`

### Retrieval Parameters

- **Semantic Retrieval**:
  - Default: k=25 candidates, top_k=12 used
  - List queries: k=40 candidates, top_k=20 used
  
- **Table Retrieval**:
  - Default: Up to 22 tables for multi-document queries
  - Calculation queries: Prioritizes base rate and factor tables
  - Split table detection: Based on PDF ID and table number patterns

- **Expansion**:
  - Max expansion: 200 additional chunks
  - Page-level: All content from same page
  - Document-level: Related content from same document

### Optional Components

- **FlashRank**: Optional reranking (skips gracefully if unavailable)
- **Service Account Key**: Auto-detects `zesty-*.json` in project root

## Key Methods

### RAGAgent

- **`answer(question: str)`**: Main entry point - processes query and returns answer with sources
- **`_semantic_retrieval(query: str, k: int)`**: Performs semantic retrieval with reranking
- **`_is_calculation_query(question: str)`**: Detects if query requires calculation
- **`_extract_calculation_inputs(question: str)`**: Extracts relevant terms for calculations

### RetrievalLogic

- **`semantic_retrieval(query: str, k: int)`**: Vector similarity search with optional reranking
- **`expand_by_page_and_document(results, max_expansion)`**: Expands retrieval to same pages/documents
- **`full_table_retrieval(query, table_ids, calculation_inputs)`**: Main table retrieval method
- **`search_tables_by_content(search_terms, ...)`**: Value-based table search
- **`search_tables_by_column_name(keywords)`**: Column-based table search
- **`group_split_tables(table_ids)`**: Detects and groups split tables
- **`merge_split_tables(table_group)`**: Merges split tables into single Markdown
- **`_find_base_rate_tables(rate_tables)`**: Dynamic base rate table discovery
- **`_find_factor_tables(query, rate_tables)`**: Dynamic factor table discovery

### Utils

- **`extract_specific_values_from_query(query)`**: Extracts ZIP codes, territories, percentages, model names
- **`extract_geographic_terms(query)`**: Extracts street names and geographic boundaries
- **`extract_table_ids_from_results(results)`**: Extracts table IDs from semantic results
- **`table_id_to_bigquery_table_name(table_id)`**: Converts metadata format to BigQuery format
- **`are_likely_splits(table1, table2, schema1, schema2)`**: Checks if tables are likely splits

## Dependencies

- **langchain-google-vertexai**: LLM (Gemini) and embeddings
- **langchain-google-community**: BigQuery Vector Store
- **google-cloud-bigquery**: BigQuery client for structured data
- **flashrank**: Optional reranking (gracefully handles if unavailable)
- **pandas**: DataFrame manipulation for table formatting

## Error Handling

- **Missing Dependencies**: Clear error messages if required packages are missing
- **Authentication Errors**: Automatic verification with helpful error messages
- **Reranking Failures**: Falls back to original order if FlashRank fails
- **Table Fetching Errors**: Continues with available tables if some fail
- **Split Table Detection**: Conservative approach - only merges when confident

## Performance Considerations

- **Parallel Execution**: Table fetching uses ThreadPoolExecutor (20 workers)
- **Batch Operations**: Schema fetching uses INFORMATION_SCHEMA for efficiency
- **Caching**: Table list cached for 60 seconds
- **Token Management**: Monitors input tokens to stay within model limits
- **Selective Retrieval**: Only retrieves tables that pass relevance checks

## Future Improvements

- **Query Classification**: Could add explicit query type classification (currently implicit)
- **Adaptive Retrieval**: Adjust retrieval parameters based on query complexity
- **Result Caching**: Cache retrieval results for repeated queries
- **Incremental Updates**: Support incremental index updates without full rebuild

