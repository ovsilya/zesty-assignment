# Table Retrieval Limits - Theoretical Maximum

## Current Implementation

### Hard Limits
- **Maximum tables retrieved**: 12 tables (hardcoded limit)
- **Rows per table**: 200 rows (LIMIT 200 in SQL query)
- **Location**: `src/retrieval/rag_agent.py` line 510: `relevant_tables[:12]`

### Current Usage
- **Semantic retrieval**: ~8-10 chunks × ~1000 chars ≈ 8,000-10,000 tokens
- **Table retrieval**: 12 tables × 200 rows × ~50 chars/row ≈ 120,000 chars ≈ 30,000 tokens
- **Prompt overhead**: ~500-1,000 tokens
- **Total input**: ~39,000-41,000 tokens
- **Output**: Up to 8,192 tokens

## Theoretical Maximum Calculation

### Token Budget
- **Gemini 2.5 Pro Context Window**: 128,000 tokens (input + output)
- **Max Output Tokens**: 8,192 tokens (configured)
- **Max Input Tokens**: ~119,808 tokens (128,000 - 8,192)

### Token Allocation
- **Semantic Retrieval**: ~10,000 tokens (8-10 chunks)
- **Prompt Overhead**: ~1,000 tokens (instructions, formatting)
- **Available for Tables**: ~108,808 tokens (119,808 - 10,000 - 1,000)

### Table Size Estimate
- **Rows per table**: 200 rows (current limit)
- **Average chars per row**: ~50 characters
- **Chars per table**: ~10,000 characters (200 × 50)
- **Tokens per table**: ~2,500 tokens (10,000 chars ÷ 4 chars/token)

### Theoretical Maximum
```
Theoretical Max Tables = Available Tokens ÷ Tokens per Table
                       = 108,808 ÷ 2,500
                       ≈ 43-44 tables
```

## Practical Considerations

### 1. Processing Time
- **Current**: 12 tables × ~1-2 seconds = ~12-24 seconds
- **Theoretical max**: 44 tables × ~1-2 seconds = ~44-88 seconds
- **Impact**: Significant latency increase

### 2. LLM Processing
- Very large contexts may reduce LLM performance
- Attention mechanism may struggle with 40+ tables
- Quality may degrade with too much information

### 3. BigQuery Limits
- **No hard limit** on number of SELECT queries
- **Query timeout**: Default 6 hours (not a concern)
- **Rate limits**: May hit API rate limits with many queries

### 4. Markdown Formatting
- Markdown conversion adds overhead
- Larger tables = more formatting time
- Memory usage increases with more tables

### 5. Relevance vs. Quantity
- More tables ≠ better answers
- Irrelevant tables add noise
- Better to retrieve fewer, highly relevant tables

## Recommended Limits

### Conservative (Current)
- **12 tables**: Good balance of coverage and performance
- **200 rows per table**: Sufficient for most queries
- **Processing time**: ~15-30 seconds total

### Moderate
- **20-25 tables**: Increased coverage without major performance hit
- **200 rows per table**: Keep current limit
- **Processing time**: ~30-50 seconds total

### Aggressive (Theoretical)
- **40-44 tables**: Maximum theoretical limit
- **200 rows per table**: Keep current limit
- **Processing time**: ~60-90 seconds total
- **Risk**: May degrade LLM performance

## Variable Factors

### Table Size
- **Small tables** (< 50 rows): Can retrieve more
- **Large tables** (> 200 rows): Current limit may be too much
- **Wide tables** (many columns): More tokens per row

### Query Complexity
- **Simple queries**: Fewer tables needed
- **Complex queries**: May benefit from more tables
- **List queries**: May need more comprehensive coverage

### Semantic Retrieval Quality
- **Good matches**: Fewer tables needed
- **Poor matches**: May need more tables as fallback

## Implementation Options

### Option 1: Increase Hard Limit
```python
# Change from 12 to 20-25
relevant_tables = relevant_tables[:25]
```

### Option 2: Adaptive Limit
```python
# Adjust based on query type
if is_list_query:
    max_tables = 20
elif is_complex_query:
    max_tables = 15
else:
    max_tables = 12
```

### Option 3: Dynamic Based on Token Budget
```python
# Calculate based on available tokens
available_tokens = estimate_available_tokens()
max_tables = min(available_tokens // TOKENS_PER_TABLE, 40)
```

### Option 4: Two-Phase Retrieval
```python
# Phase 1: Retrieve top 12 tables
# Phase 2: If answer incomplete, retrieve additional 12
```

## Current Test Results

With **12 tables**:
- **EF_1**: 22.86% (list query - may benefit from more)
- **EF_2**: 29.39% (found relevant data)
- **EF_3**: 0.00% (found factor but not base rate)

## Conclusion

**Theoretical Maximum**: ~40-44 tables (with 200 rows each)

**Practical Recommendation**: 
- **Current (12 tables)**: Good for most queries
- **Moderate (20-25 tables)**: Better for complex queries
- **Maximum (40 tables)**: Only for very complex queries, with performance trade-offs

**Key Constraint**: Not token limits, but:
1. Processing time
2. LLM attention/quality with very large contexts
3. Relevance (more tables ≠ better answers)

