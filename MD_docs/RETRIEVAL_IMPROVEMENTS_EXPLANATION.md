# RAG Retrieval Improvements - Universal Approach

## Problem Identified

The initial RAG agent was missing 10 out of 33 expected rating plan rules (69.7% coverage). Investigation revealed that **all missing items existed in the vector store** but were in different documents/pages that weren't retrieved by the initial semantic search.

## Root Cause

1. **Limited Initial Retrieval**: The semantic search for "List all rating plan rules" only retrieved 15-20 top results
2. **Incomplete Coverage**: Many rules were mentioned in documents that weren't in the top semantic matches
3. **No Follow-up Search**: The system didn't perform additional targeted searches to find related items

## Changes Made (Universal Approach)

### 1. Increased Retrieval Scope for "List All" Queries

**Location**: `src/retrieval/rag_agent.py`, lines 997-999

```python
# Before:
k_retrieve = 30 if is_list_query else 25
top_k_use = 15 if is_list_query else 12

# After:
k_retrieve = 40 if is_list_query else 25  # Increased from 30 to 40
top_k_use = 20 if is_list_query else 12   # Increased from 15 to 20
```

**Why**: "List all" queries require more comprehensive retrieval to find all items, not just the most relevant ones.

### 2. Universal Key Term Extraction and Additional Searches

**Location**: `src/retrieval/rag_agent.py`, lines 1007-1038

**What Changed**: 
- **Removed**: Hardcoded list of 10 specific rating rule names (question-specific)
- **Added**: Universal key term extraction that works for any "list all" query

**How It Works**:

1. **Extract Key Terms from Initial Results**:
   - Analyzes top 10 semantic retrieval results
   - Uses regex to find capitalized phrases (potential rule/item names)
   - Filters for meaningful phrases (2+ words, >10 characters)
   - Skips common words ("the", "and", "for", etc.)

2. **Extract Key Terms from Query**:
   - Finds capitalized terms in the query itself
   - Identifies two-word capitalized phrases

3. **Perform Additional Targeted Searches**:
   - For each extracted key term, performs a semantic search
   - Retrieves top 1 result per term (to avoid duplicates)
   - Limits to top 15 terms to avoid too many searches

4. **Merge Results**:
   - Adds new results to the initial retrieval
   - Uses content hash to avoid duplicates

**Example**:
- Query: "List all rating plan rules"
- Initial results contain: "Rule C-1: Limits of Liability"
- System extracts: "Limits of Liability", "Coverage Relationships", etc.
- Performs additional searches on these terms
- Finds related rules that weren't in initial top results

### 3. Increased Page/Document Expansion Limit

**Location**: `src/retrieval/rag_agent.py`, line 1042

```python
# Before:
expanded_sem = self._expand_by_page_and_document(top_sem, max_expansion=200)

# After:
expanded_sem = self._expand_by_page_and_document(top_sem, max_expansion=300)
```

**Why**: When a relevant chunk is found, we want to retrieve ALL content from the same page/document to ensure completeness.

## Why This Approach is Universal

1. **No Hardcoded Lists**: Doesn't rely on question-specific terms
2. **Works for Any "List All" Query**: 
   - "List all rating plan rules" → extracts rule names
   - "List all coverage types" → extracts coverage names
   - "List all discounts" → extracts discount names
3. **Adaptive**: Extracts terms from the actual retrieved content, not from a predefined list
4. **Scalable**: Works regardless of domain or question type

## Benefits

1. **Better Coverage**: Finds items that exist in the index but weren't in top semantic matches
2. **Domain Agnostic**: Works for any "list all" query, not just rating rules
3. **Maintainable**: No need to update code when new question types are added
4. **Efficient**: Limits additional searches to meaningful terms (max 15)

## Trade-offs

1. **Latency**: Additional searches add ~2-5 seconds for "list all" queries
2. **Token Usage**: More content retrieved = higher token usage
3. **Precision vs Recall**: Prioritizes recall (finding all items) over precision (only most relevant)

## Testing Results

After improvements:
- **Retrieved**: 199 semantic chunks (up from 177)
- **Tables**: 32 tables (up from 26)
- **Pages**: 16 pages (up from 10)
- **Files**: 6 files (up from 5)
- **Answer Length**: 9,499 characters (more comprehensive)

The system now finds more rules including:
- Rule C-23: Trampoline Factor
- Rule C-24: Roof Condition Factor
- Rule C-25: Tree Overhang Factor
- Rule C-26: Solar Panel Factor
- Rule C-29: Secondary Heating Source Factor
- Rule C-31: Endorsement Combination Discount
- Rule C-33: Claims Free Discount
- Rule C-35: Minimum Premium

## Future Improvements

1. **LLM-based Term Extraction**: Use the LLM to extract key terms from the query and initial results (more intelligent)
2. **Iterative Refinement**: After generating initial answer, check for missing items and do additional searches
3. **Confidence Scoring**: Rank additional search results by relevance to avoid noise

