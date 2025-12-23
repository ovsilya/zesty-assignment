# Keyword Matching in the RAG Pipeline - Explanation

## Overview

The RAG pipeline uses a **three-tier priority system** to find relevant tables from BigQuery. Keywords are used as fallback mechanisms when semantic search doesn't find enough tables or misses relevant ones.

## The Three-Tier Priority System

### Priority 1: Semantic Search (Primary - Best Quality)
**How it works:**
1. User asks a question
2. Query is embedded using Vertex AI embeddings
3. Vector search finds similar content in the vector index
4. System extracts `table_id` values from table summaries found in results
5. Maps those table IDs to BigQuery table names

**Why it's best:**
- Uses LLM-generated table summaries (semantic understanding)
- Understands query intent, not just keywords
- Finds tables based on meaning, not just text matching
- Example: Query "hurricane premium calculation" finds tables about "hurricane deductibles" even if the word "premium" isn't in the table name

**Limitations:**
- Depends on quality of table summaries
- May miss tables if summaries don't capture all relevant aspects
- Limited by how many table summaries are in the top K results
- If semantic search finds 0-2 tables, we need more coverage

### Priority 2: Column-Based Search (Secondary - Precise Matching)
**How it works:**
1. Extract keywords from the query that relate to column names
2. Search all BigQuery tables for columns matching those keywords
3. Score tables by how many matching columns they have
4. Select top-scoring tables

**Example keywords extracted:**
- Query: "hurricane deductible factor"
  - Keywords: `["hurricane_deductible", "mandatory_hurricane"]`
- Query: "territory rate change"
  - Keywords: `["territory", "zip_code", "rate"]`
- Query: "motorcycle GRG"
  - Keywords: `["grg", "collision_rating_group", "motorcycle", "mcy"]`

**Why it's needed:**
- **Precision**: Finds tables with specific column names (e.g., "hurricane_deductible")
- **Complements semantic**: Semantic might find "hurricane" tables, but column search finds tables with exact column names
- **Handles technical terms**: GRG, DRG, LRG are technical terms that column search handles well
- **Fills gaps**: When semantic search finds few tables, column search adds more

**Limitations:**
- Requires exact or near-exact keyword matches
- May miss tables with different naming conventions
- Doesn't understand synonyms (e.g., "premium" vs "rate")

### Priority 3: Table Name Keyword Matching (Tertiary - Broad Coverage)
**How it works:**
1. Extract general keywords from the query
2. Search table names (not columns) for these keywords
3. Match if any keyword appears in the table name

**Example:**
- Query: "hurricane premium calculation"
  - Keywords: `["hurricane", "ho3", "homeowner", "maps", "rate"]`
- Table: `table__215004905_180407973__CT_Homeowners_MAPS_Rate_Pages_Eff_8_18_25_v3_table_119`
  - Matches: "homeowner", "maps", "rate" → **Included**

**Why it's needed:**
- **Broad coverage**: Finds tables by PDF/document name patterns
- **Fallback**: When semantic and column search find few tables
- **Document-level matching**: Finds tables from relevant documents
- **Simple and fast**: No schema inspection needed

**Limitations:**
- Very broad matching (may include irrelevant tables)
- Depends on table naming conventions
- May miss tables with different naming patterns

## Why We Need Keywords

### Problem: Semantic Search Alone Isn't Enough

**Scenario 1: Limited Table Summaries**
- Semantic search retrieves top 10 chunks
- Only 1-2 of those chunks are table summaries
- Result: Only 1-2 tables identified
- **Solution**: Keywords find additional relevant tables

**Scenario 2: Summary Quality**
- Table summary says "This table shows rate information"
- Doesn't mention "hurricane" or "deductible" specifically
- Semantic search might not rank it highly
- **Solution**: Column search finds it by column name "hurricane_deductible"

**Scenario 3: Technical Terms**
- Query mentions "GRG" (Collision Rating Group)
- Table summary might say "motorcycle rating groups"
- Semantic search might miss the connection
- **Solution**: Column search finds tables with "GRG" column

**Scenario 4: New/Unseen Queries**
- Query uses terminology not in training/summaries
- Semantic embeddings might not match well
- **Solution**: Keywords provide exact matching fallback

### The Numbers

**Current System:**
- **1,227 tables** in BigQuery
- **Semantic search**: Finds 1-8 tables (from table summaries)
- **Column search**: Finds 11-26 additional tables (from 247-278 candidates)
- **Keyword matching**: Finds 0-10 additional tables (fallback)
- **Total**: Up to 27 tables retrieved

**Without Keywords:**
- Only semantic search: 1-8 tables
- Missing: 19-26 potentially relevant tables
- Result: Lower accuracy, missing data

## How Keywords Are Extracted

### From Query Text
The system extracts keywords by pattern matching:

```python
# Example: "hurricane deductible factor"
if "hurricane" in query and "deductible" in query:
    keywords = ["hurricane_deductible", "mandatory_hurricane"]

# Example: "territory rate change"
if "territory" in query:
    keywords = ["territory", "zip_code"]
if "rate" in query:
    keywords.extend(["rate", "rating"])
```

### Two Types of Keywords

1. **Column Keywords** (Priority 2):
   - More specific: `"base_rate"`, `"ho3_a_rate"`, `"coverage_a"`
   - Used to search column names
   - Example: Finds tables with column named "hurricane_deductible"

2. **Table Name Keywords** (Priority 3):
   - More general: `"hurricane"`, `"rate"`, `"territory"`
   - Used to search table names
   - Example: Finds tables with "hurricane" in the name

## Current Implementation

### Priority Flow

```
Query: "Calculate hurricane premium for HO3 policy"

Step 1: Semantic Search
  → Finds 1 table (from table summary)
  
Step 2: Column Search (if < 27 tables)
  → Extracts: ["hurricane_deductible", "mandatory_hurricane", "ho3", "coverage_a"]
  → Searches 1,227 tables for columns matching these
  → Finds 26 additional tables
  
Step 3: Keyword Matching (if < 27 tables)
  → Extracts: ["hurricane", "ho3", "homeowner", "maps", "rate"]
  → Searches table names
  → Finds 0 additional tables (already have 27)
  
Result: 27 tables retrieved
```

### Code Location

**Priority 2 (Column Search)**: `src/retrieval/rag_agent.py` lines 416-472
- Extracts column-related keywords
- Calls `_search_tables_by_column_name()`
- Scores and ranks matches

**Priority 3 (Table Name Matching)**: `src/retrieval/rag_agent.py` lines 474-503
- Extracts general keywords
- Matches against table names
- Simple substring matching

## Benefits of This Approach

1. **Comprehensive Coverage**: Finds tables semantic search might miss
2. **Precision**: Column search finds exact matches for technical terms
3. **Fallback Safety**: Always finds some tables, even if semantic fails
4. **Scalability**: Works with 1,227 tables (would be impossible to manually select)
5. **Complementary**: Each method finds different types of tables

## Limitations & Trade-offs

### Limitations
1. **Keyword extraction is rule-based**: May miss synonyms or variations
2. **No semantic understanding**: "premium" and "rate" treated as different
3. **False positives**: May include irrelevant tables
4. **Maintenance**: Need to update keyword rules for new domains

### Trade-offs
- **More tables = Better coverage** but also **more noise**
- **Keywords = Fast and precise** but **less flexible than semantic**
- **Three-tier = Comprehensive** but **adds complexity**

## Alternative Approaches (Not Implemented)

### 1. LLM-Based Keyword Extraction
- Use LLM to extract keywords from query
- More flexible, understands synonyms
- **Cost**: Additional LLM call per query

### 2. Embedding-Based Table Search
- Embed all table names/columns
- Vector search for similar tables
- **Cost**: Need to maintain embeddings for all tables

### 3. Table Catalog/Index
- Pre-build index of table purposes
- Search index instead of tables directly
- **Cost**: Maintenance overhead

## Conclusion

**Keywords are needed because:**
1. Semantic search alone finds too few tables (1-8 out of 1,227)
2. Table summaries may not capture all relevant aspects
3. Technical terms need exact matching (GRG, territory codes)
4. Provides fallback when semantic search fails
5. Complements semantic search for comprehensive coverage

**The three-tier system ensures:**
- **Quality**: Semantic search finds best matches
- **Precision**: Column search finds exact matches
- **Coverage**: Keyword matching ensures we find enough tables
- **Reliability**: Always finds relevant tables, even if one method fails

Without keywords, the system would only retrieve 1-8 tables on average, missing 19-26 potentially relevant tables and significantly reducing accuracy.

