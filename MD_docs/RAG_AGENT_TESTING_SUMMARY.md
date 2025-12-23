# RAG Agent Testing Summary

## Overview

This document summarizes the initial testing of the RAG agent using questions from `artifacts/questions.csv`. The agent uses a hybrid retrieval approach combining:
- **Semantic retrieval** from the vector index (DOC Index) for conceptual questions
- **SQL retrieval** from the facts store (BigQuery tables) for quantitative questions

## Test Questions

Three questions were tested:

1. **EF_1**: "List all rating plan rules" (semantic query)
2. **EF_2**: Territory/Comprehensive rate change + Motorcycle rating groups comparison (quantitative query)
3. **EF_3**: Hurricane premium calculation (quantitative query)

## Current Results

### Overall Performance
- **Total Questions**: 3
- **Exact Matches**: 0 (0.0%)
- **Average Overall Score**: 16.43%
- **Average Keyword Coverage**: 22.55%
- **Average Number Match**: 37.04%

### Per-Question Breakdown

#### EF_1: List all rating plan rules
- **Category**: Semantic
- **Score**: 23.43%
- **Status**: ✗
- **Issue**: Semantic retrieval found some rules (e.g., "Rating Perils", "Loss History Rating") but not the comprehensive list expected. The vector index may need more comprehensive indexing of rule lists, or the query needs to retrieve more chunks.

#### EF_2: Territory/Comprehensive rate + Motorcycle GRG comparison
- **Category**: Quantitative
- **Score**: 25.86%
- **Status**: ✗
- **Issue**: SQL generation failed to find the correct tables or columns. With 1,227 tables in BigQuery, the LLM struggled to identify the relevant tables for:
  - State Farm Fire & Casualty Company rate disruption data
  - Territory 117/118 for Zip Code 89110
  - Motorcycle rating groups (Ducati Panigale V4 R, Honda Grom ABS)

#### EF_3: Hurricane premium calculation
- **Category**: Quantitative
- **Score**: 0.00%
- **Status**: ✗
- **Issue**: SQL query returned "No matching rows found". The query likely:
  - Didn't find the correct base rate table for HO3 policies
  - Didn't locate the hurricane deductible factor table
  - Used incorrect column names or filters

## Improvements Made

### 1. Always Use Hybrid Approach ✅
- **Removed query classification**: Agent now always performs both semantic and quantitative retrieval
- **Benefits**: Comprehensive coverage for all query types

### 2. Table Identification via Semantic Search ✅
- **Two-stage retrieval**: 
  1. Semantic retrieval finds relevant table summaries
  2. Extract `table_id` values from table summary metadata
  3. Use those table IDs to identify corresponding BigQuery tables
- **Implementation**: 
  - `_extract_table_ids_from_results()`: Extracts table IDs from semantic results
  - `_find_bigquery_table_by_table_id()`: Maps table_id to BigQuery table name
  - `_get_relevant_table_schemas()`: Prioritizes tables from semantic search
- **Benefits**: More accurate table selection using LLM-generated table summaries

### 3. Schema-Aware SQL Generation
- Enhanced `_sql_retrieval()` to fetch actual BigQuery table schemas (column names and types)
- Provides schema information to the LLM so it can generate SQL with correct column names
- Added retry logic with error feedback for failed SQL queries

### 4. Handling Generic Column Names (col1, col2, etc.) ✅
- **Problem**: Some tables have generic column names when original headers were missing
- **Solution**: 
  - Added warnings in SQL generation prompts about generic column names
  - Instructed LLM to check if first row contains headers
  - Added guidance in retry prompts for handling this issue

### 5. Enhanced Semantic Retrieval
- Increased retrieval count: Default 15 (up from 10), List queries 20 (up from 10)
- Increased results used: Default 8 (up from 5), List queries 10 (up from 5)
- Better handling of comprehensive list-type questions

### 6. Improved SQL Prompts
- Added guidance about column name sanitization (underscores instead of spaces)
- Included information about metadata columns (meta_source_pdf, meta_table_id, meta_page_number)
- Added notes about data type handling (STRING columns may need CAST for numeric operations)
- **New**: Warnings about generic column names (col1, col2) and first row potentially containing headers

### 7. Enhanced Answer Generation
- Explicitly mentions both semantic text and structured table data
- Special handling for "list all" queries with comprehensive extraction instructions
- Better source attribution (includes table IDs when available)

## Current Architecture (Updated)

```
Query
  ↓
Step 1: Semantic Retrieval (always performed)
  ├─→ Vector search → Rerank → Top K chunks
  └─→ Extract table_ids from table_summary results
  ↓
Step 2: SQL Retrieval (always performed)
  ├─→ Use table_ids from semantic search (priority)
  ├─→ Fallback to keyword matching
  ├─→ Get schemas for identified tables
  └─→ Generate SQL → Execute → Retry on error
  ↓
Step 3: Combine Contexts
  ├─→ Semantic text chunks
  └─→ Table data from SQL
  ↓
Step 4: LLM Answer Generation
  └─→ Comprehensive answer using both sources
```

**Key Change**: Always uses hybrid approach (both semantic + quantitative) for all queries.

## Identified Issues & Next Steps

### Issue 1: Table Selection for SQL Queries ✅ IMPLEMENTED
**Problem**: With 1,227 tables, the LLM struggles to identify the right tables even with keyword filtering.

**Solution Implemented**:
- ✅ **Two-stage retrieval**: First use semantic search to find relevant table summaries, then extract table IDs for SQL generation
- ✅ **Table ID mapping**: Maps table_id from metadata to BigQuery table names
- ✅ **Priority-based selection**: Tables from semantic search are prioritized over keyword matching

**Remaining Work**:
- Monitor if table_id to BigQuery name mapping is accurate for all cases
- Consider adding table catalog for common query patterns

### Issue 2: Comprehensive List Queries ✅ IMPROVED
**Problem**: "List all rating plan rules" doesn't retrieve all rules comprehensively.

**Solutions Implemented**:
- ✅ **Hybrid approach**: Always use both semantic retrieval AND SQL to find rule lists
- ✅ **Increased retrieval**: More candidates retrieved and used for list queries
- ✅ **Enhanced prompts**: Explicit instructions for comprehensive extraction in list queries

**Remaining Work**:
- Monitor if table summaries include comprehensive lists
- Consider hierarchical retrieval for very long lists

### Issue 3: Column Name Mismatches ✅ IMPROVED
**Problem**: SQL queries use incorrect column names even with schema information. Some tables have generic names (col1, col2) when headers were missing.

**Solutions Implemented**:
- ✅ **Schema awareness**: Provides actual column names and types to LLM
- ✅ **Generic column handling**: Warnings and guidance about col1/col2 tables
- ✅ **First row header detection**: Instructions to check if first row contains headers
- ✅ **Retry logic**: Enhanced error handling with specific guidance

**Remaining Work**:
- Monitor SQL query success rate
- Consider column name mapping for common patterns
- Add fuzzy matching for column names

### Issue 4: Data Type Handling
**Problem**: Numeric values stored as STRING require explicit casting, which the LLM doesn't always do.

**Potential Solutions**:
1. **Automatic type inference**: Detect numeric columns and suggest CAST operations
2. **Schema enhancement**: Include data type hints in the prompt more prominently
3. **Query validation**: Pre-validate SQL queries and suggest fixes before execution

## Recommendations

### Short-term (Immediate) ✅ MOSTLY COMPLETE
1. ✅ **Table identification via semantic search**: Implemented two-stage retrieval
2. ✅ **Improved error handling**: Enhanced retry logic with specific guidance
3. ✅ **Enhanced semantic retrieval prompts**: Special handling for list queries
4. **Remaining**: Add table metadata search by PDF name for additional filtering

### Medium-term (Next Phase)
1. **Build table catalog**: Create a searchable index of table purposes and contents
2. **Implement two-stage SQL**: Semantic search → Table IDs → SQL generation
3. **Add query examples**: Include successful query examples in prompts for similar question types

### Long-term (Evaluation Phase)
1. **Fine-tune retrieval**: Use RAGAS or similar to evaluate and improve retrieval quality
2. **Agentic routing**: Implement more sophisticated routing that can break complex questions into sub-queries
3. **Feedback loop**: Learn from successful queries to improve future SQL generation

## Test Artifacts

- **Results CSV**: `test_results/rag_test_results.csv`
- **Summary Report**: `test_results/rag_test_summary.txt`
- **Diagnostic Script**: `diagnose_bigquery.py` (for inspecting BigQuery schemas)

## Conclusion

The RAG agent infrastructure is functional:
- ✅ Query classification works
- ✅ Semantic retrieval retrieves relevant chunks
- ✅ SQL generation attempts to query BigQuery
- ✅ Answer synthesis combines contexts

However, accuracy is low due to:
- Table selection challenges with 1,227 tables
- Column name mismatches in SQL queries
- Incomplete retrieval for comprehensive list queries

The improvements made (schema awareness, table filtering, enhanced retrieval) provide a foundation, but further optimization is needed for production-quality results.

