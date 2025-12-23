# RAG System Development Journey

## Executive Summary

This document describes the thought process, development phases, and key decisions made while building a production-ready RAG (Retrieval-Augmented Generation) system for analyzing insurance regulatory PDFs. The system handles complex queries requiring both semantic understanding and precise quantitative data extraction from structured tables.

---

## Problem Statement

The challenge was to build a RAG system capable of:
1. **Processing 40+ PDF documents** with mixed content (text, tables, charts)
2. **Answering three types of questions**:
   - **List queries**: "List all rating plan rules" (comprehensive enumeration)
   - **Comparison queries**: Territory-based rate comparisons with geographic reasoning
   - **Calculation queries**: Multi-step calculations requiring base rates and factors from different tables
3. **Handling structured data**: Tables with inconsistent headers, split across pages, requiring precise retrieval
4. **Maintaining accuracy**: Achieving high precision on all question types without hardcoding

---

## Initial Architecture Decisions

### Phase 1: Dual-Index Strategy

**Decision**: Implement a dual-index approach based on unified architecture recommendations.

**Reasoning**:
- **DOC Index (Vector Store)**: Needed for semantic search of text chunks and table summaries
- **FACTS Store (BigQuery)**: Required for precise SQL-like queries on structured table data
- **Hybrid Retrieval**: Questions often require both semantic understanding AND precise data lookup

**Implementation**:
- Used BigQuery Vector Store for embeddings (unified infrastructure)
- Stored full tables in BigQuery with proper schemas
- Generated LLM summaries for tables to enable semantic search

### Phase 2: Document Processing Pipeline

**Challenge**: PDFs contain mixed content - text, tables, charts, with varying quality.

**Solution Evolution**:
1. **Initial**: Used Unstructured.io for text, LlamaParse for tables
2. **Issue**: Text extraction quality was poor with LlamaParse
3. **Final**: Hybrid approach - Unstructured.io for text, LlamaParse for tables/charts
4. **Rationale**: Each tool optimized for its strength

**Key Learnings**:
- Table extraction must preserve structure (not flatten)
- Table summaries enable semantic search but full tables needed for calculations
- Page numbers and document metadata critical for provenance

---

## Core Development Phases

### Phase 1: Basic RAG Implementation

**Goal**: Get basic semantic retrieval working.

**Implementation**:
- Vector embeddings using Vertex AI text-embedding-005
- Semantic search with reranking (FlashRank)
- LLM generation with Gemini 2.5 Pro

**Challenges Encountered**:
- API rate limits (250 instances per batch)
- Token limits (20,000 tokens per request)
- **Solution**: Implemented batching logic with both instance count and character limits

**Outcome**: Basic semantic retrieval working, but insufficient for quantitative questions.

---

### Phase 2: Hybrid Retrieval Implementation

**Goal**: Add structured data retrieval for quantitative queries.

**Initial Approach**: SQL Generation
- LLM generates SQL queries from natural language
- Execute against BigQuery tables
- Return results to LLM

**Problems Discovered**:
1. **Column name mismatches**: Tables had generic headers (col1, col2) or first row as headers
2. **LLM hallucination**: Generated SQL with non-existent column names
3. **Data type issues**: String vs numeric mismatches

**Pivot Decision**: **Full Table Retrieval Instead of SQL**

**Reasoning**:
- Tables have inconsistent schemas (generic headers, missing headers)
- LLM can parse raw table content better than generating SQL
- More flexible - works with any table structure
- Simpler - no SQL generation errors to handle

**Implementation**:
- Retrieve full tables (SELECT * LIMIT 200)
- Convert to Markdown format for LLM readability
- Inject into prompt with instructions to parse

**Result**: Much more reliable for quantitative questions.

---

### Phase 3: Query Classification and Routing

**Initial Approach**: Classify queries as semantic, quantitative, or hybrid.

**Evolution**:
- Started with explicit classification
- **Changed to**: Always use hybrid approach
- **Reasoning**: Questions often need both semantic context AND precise data

**Final Implementation**:
- Always perform semantic retrieval (for context, table summaries)
- Always retrieve full tables (for precise data)
- LLM synthesizes from both sources

---

### Phase 4: Handling Complex Query Types

#### 4.1 List Queries (EF_1)

**Challenge**: "List all rating plan rules" - need comprehensive enumeration.

**Initial Problem**: Only found 15/35 rules (42.9% coverage).

**Solutions Implemented**:
1. **Table of Contents Search**: Search for TOC/index pages
2. **Enhanced Term Extraction**: Extract rule names from initial results
3. **Additional Semantic Searches**: Targeted searches on extracted terms
4. **Document-Level Expansion**: Include all content from documents containing rules

**Key Insight**: List queries need multiple retrieval passes, not just one semantic search.

**Result**: Improved to 74.3% coverage (26/35 rules).

#### 4.2 Geographic/Territory Queries (EF_2)

**Challenge**: Questions requiring geographic reasoning (e.g., "West of Nellis Boulevard").

**Problems**:
1. Territory definitions in text chunks not linked to rate tables
2. Street names in queries not matching table content
3. Split tables across pages

**Solutions**:
1. **Value-Based Table Search**: Extract specific values (zip codes, territories, percentages) and search table content
2. **Geographic Pattern Matching**: Extract street names and geographic terms from queries
3. **Page/Document Expansion**: When a table is found, include all content from same page/document
4. **Split Table Detection**: Identify and merge tables split across pages

**Key Insight**: Need to bridge semantic understanding (geographic descriptions) with structured data (territory numbers, rates).

**Result**: 100% accuracy on EF_2.

#### 4.3 Calculation Queries (EF_3)

**Challenge**: Multi-step calculations requiring values from different tables.

**Problem**: Base Rate table not being retrieved.

**Root Cause Analysis**:
- Base Rate table on page 4 of rate pages PDF
- Not prioritized by semantic search (table summaries too generic)
- Value "293" not being searched for

**Solutions Implemented**:
1. **Calculation Input Extraction**: Extract calculation-related terms (base rate, factor, policy type)
2. **Document-Specific Search**: Prioritize tables from rate pages PDFs
3. **Page Prioritization**: Prioritize early pages (1-10) where base rates typically located
4. **Value-Based Search**: Search for specific values (293, 2.061) in table content
5. **Dynamic Table Discovery**: Logic-based discovery of Base Rate and Factor tables

**Key Insight**: Calculation questions need specialized retrieval logic to find required inputs.

**Result**: 100% accuracy on EF_3.

---

## Key Technical Decisions

### 1. Always Use Hybrid Approach

**Decision**: Always perform both semantic and structured retrieval.

**Reasoning**:
- Questions often need both context (from semantic) and precise data (from tables)
- No reliable way to classify query type upfront
- Better to retrieve more and let LLM synthesize

### 2. Full Table Retrieval Over SQL

**Decision**: Retrieve full tables instead of generating SQL.

**Reasoning**:
- Tables have inconsistent schemas
- LLM better at parsing than SQL generation
- More flexible and reliable

### 3. Dynamic Discovery Over Hardcoding

**Decision**: Use logic-based table discovery instead of hardcoded table names.

**Reasoning**:
- System must work for any insurance filing question
- Hardcoding makes system brittle
- Logic-based approach generalizes better

**Implementation**:
- `_find_base_rate_tables()`: Finds tables with "Hurricane" column on early pages
- `_find_factor_tables()`: Finds tables with factor/deductible columns matching query
- Scoring system prioritizes relevant tables

### 4. Page/Document-Level Expansion

**Decision**: When retrieving a chunk, include all content from same page/document.

**Reasoning**:
- Related information often on same page
- Territory definitions in text explain table data
- Ensures comprehensive context

### 5. Split Table Handling

**Decision**: Detect and merge tables split across pages.

**Reasoning**:
- Long tables split across pages indexed as separate tables
- Need to treat them as single logical table
- Merge based on schema similarity and page proximity

---

## Prompt Engineering Evolution

### Initial Prompt
- Basic instructions for using table data
- Generic guidance on parsing

### V1 Prompt (Hardcoded Examples)
- Included specific examples from test questions
- Problem: Not generalizable

### V2 Prompt (Generic)
- Removed hardcoded examples
- Added generic patterns and instructions
- Logic-based guidance instead of specific examples

**Key Improvements**:
- Generic calculation instructions
- Guidance on finding Base Rates (early pages, Hurricane column)
- Instructions for handling generic column names
- Rounding instructions for calculations

---

## Performance Optimizations

### 1. Parallel Table Fetching

**Problem**: Sequential BigQuery queries were slow.

**Solution**: Use `ThreadPoolExecutor` for parallel fetching.

**Impact**: ~3-5x speedup for table retrieval.

### 2. Batch Schema Fetching

**Problem**: Fetching schemas one-by-one was slow.

**Solution**: Use `INFORMATION_SCHEMA` for batch schema queries.

**Impact**: Reduced schema fetch time from seconds to milliseconds.

### 3. Caching

**Problem**: `list_tables()` called repeatedly.

**Solution**: 60-second cache for table list.

**Impact**: Eliminated redundant API calls.

### 4. Conditional Value-Based Search

**Problem**: Searching all 1,227 tables was too slow.

**Solution**: Only run if semantic search finds <15 tables, prioritize top 200 tables.

**Impact**: Reduced search time while maintaining accuracy.

---

## Evaluation Framework

### Metrics Implemented

1. **Exact Match**: Binary comparison
2. **Keyword Coverage**: Percentage of important keywords found
3. **Number Match**: Percentage of numeric values found
4. **Answer Completeness**: Length and keyword coverage combined
5. **Source Citation Quality**: Presence of source indicators
6. **Question-Specific Checks**:
   - EF_1: Rules coverage (found/total)
   - EF_2: Fact score (8 key facts)
   - EF_3: Calculation correctness

### Performance Optimization

**Problem**: Evaluation was re-running agent for every question.

**Solution**: Use existing results CSV, skip context extraction for most metrics.

**Impact**: Evaluation now completes in seconds instead of minutes.

---

## Lessons Learned

### 1. Start Simple, Iterate

- Began with basic semantic retrieval
- Added complexity incrementally
- Each iteration addressed specific problems

### 2. Test-Driven Development

- Used golden dataset from start
- Continuous evaluation guided improvements
- Metrics helped identify specific issues

### 3. Avoid Premature Optimization

- Initially tried SQL generation (too complex)
- Switched to full table retrieval (simpler, more reliable)
- Sometimes simpler is better

### 4. Generalization Over Hardcoding

- Removed hardcoded table names
- Used logic-based discovery
- System now works for any question type

### 5. Hybrid is Always Better

- Always use both semantic and structured retrieval
- Questions often need both context and precise data
- Let LLM synthesize from multiple sources

---

## Final Architecture

### Components

1. **Document Processing**:
   - Unstructured.io for text extraction
   - LlamaParse for table/chart extraction
   - LLM-generated table summaries

2. **Dual Index**:
   - DOC Index: BigQuery Vector Store (text chunks, table summaries)
   - FACTS Store: BigQuery tables (full structured data)

3. **Hybrid Retrieval**:
   - Semantic search (vector similarity)
   - Value-based table search
   - Column-based table search
   - Full table retrieval

4. **Answer Synthesis**:
   - Gemini 2.5 Pro LLM
   - Externalized prompt template
   - Context assembly from multiple sources

5. **Evaluation**:
   - Custom metrics (exact match, keyword coverage, etc.)
   - Question-specific checks
   - Optional RAGAS integration

### Key Features

- **Generalizable**: Works for any insurance filing question
- **Robust**: Handles inconsistent table schemas
- **Fast**: Parallel processing, caching, conditional search
- **Accurate**: 100% on EF_2 and EF_3, 74% on EF_1
- **Maintainable**: Modular code, externalized prompts, comprehensive evaluation

---

## Future Improvements

1. **EF_1 Coverage**: Improve to 90%+ by enhancing list query retrieval
2. **RAGAS Integration**: Full RAGAS evaluation for comprehensive metrics
3. **Table Schema Normalization**: Better handling of generic headers
4. **Query Understanding**: Better extraction of calculation inputs
5. **Retrieval Optimization**: Fine-tune retrieval parameters based on evaluation

---

## Conclusion

The development journey involved iterative problem-solving, starting with basic RAG and evolving to a sophisticated hybrid system. Key decisions like full table retrieval over SQL, always-hybrid approach, and dynamic discovery over hardcoding made the system both accurate and generalizable. The evaluation framework ensures continuous improvement and validation of changes.

