# Keyword Workflow - Visual Diagram

## The Three-Tier Table Selection Process

```
┌─────────────────────────────────────────────────────────────┐
│                    USER QUERY                               │
│  "Calculate hurricane premium for HO3 policy"               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         TIER 1: SEMANTIC SEARCH (Primary)                   │
│  ────────────────────────────────────────────────────────   │
│  1. Embed query → Vector search in doc_index                │
│  2. Find table summaries with similar meaning               │
│  3. Extract table_id from metadata                          │
│  4. Map to BigQuery table names                             │
│                                                             │
│  Result: 1-8 tables found                                   │
│  Quality: ★★★★★ (Best - semantic understanding)             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    Need more tables?
                            │
                    ┌───────┴───────┐
                    │  Yes (< 27)   │
                    └───────┬───────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│      TIER 2: COLUMN-BASED SEARCH (Secondary)                │
│  ────────────────────────────────────────────────────────   │
│  1. Extract column keywords from query:                     │
│     ["hurricane_deductible", "mandatory_hurricane",         │
│      "ho3", "coverage_a"]                                   │
│                                                             │
│  2. Search all 1,227 tables for columns matching keywords   │
│                                                             │
│  3. Score tables by number of matching columns              │
│                                                             │
│  4. Select top-scoring tables                               │
│                                                             │
│  Result: +11-26 additional tables                           │
│  Quality: ★★★★☆ (Good - precise matching)                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    Still need more?
                            │
                    ┌───────┴───────┐
                    │  Yes (< 27)   │
                    └───────┬───────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│    TIER 3: TABLE NAME KEYWORD MATCHING (Tertiary)           │
│  ────────────────────────────────────────────────────────   │
│  1. Extract general keywords from query:                    │
│     ["hurricane", "ho3", "homeowner", "maps", "rate"]       │
│                                                             │
│  2. Search table names (not columns) for keywords           │
│                                                             │
│  3. Match if any keyword appears in table name              │
│                                                             │
│  Result: +0-10 additional tables                            │
│  Quality: ★★★☆☆ (Fair - broad matching)                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    FINAL RESULT                             │
│  ────────────────────────────────────────────────────────   │
│  Up to 27 tables retrieved:                                 │
│  • 1-8 from semantic search                                 │
│  • 11-26 from column search                                 │
│  • 0-10 from keyword matching                               │
│                                                             │
│  → Detect split tables → Merge → Send to LLM                │
└─────────────────────────────────────────────────────────────┘
```

## Why Each Tier is Needed

### Tier 1: Semantic Search
**Finds**: Tables based on meaning and context
**Example**: Query "hurricane premium" finds tables about "hurricane deductibles" even if "premium" isn't mentioned
**Why needed**: Best quality matches, understands intent

### Tier 2: Column Search  
**Finds**: Tables with specific column names
**Example**: Query "GRG" finds tables with "GRG" or "collision_rating_group" columns
**Why needed**: 
- Semantic might miss technical terms
- Finds tables with exact column matches
- Fills gaps when semantic finds few tables

### Tier 3: Keyword Matching
**Finds**: Tables by document/PDF name patterns
**Example**: Query "hurricane" finds tables from "Hurricane_Rate_Pages.pdf"
**Why needed**:
- Fallback when other methods find few tables
- Finds tables from relevant documents
- Broad coverage

## Real Example from Test Results

### EF_2 Query: "Territory/Comprehensive rate + Motorcycle GRG"

**Tier 1 (Semantic)**:
- Found: 8 tables from table summaries
- Quality: High (semantic understanding)

**Tier 2 (Column Search)**:
- Keywords extracted: `["territory", "zip_code", "comprehensive", "grg", "collision_rating_group", "motorcycle", "mcy"]`
- Searched: 1,227 tables
- Found: 247 tables with matching columns
- Selected: Top 19 (scored by number of matching columns)
- Quality: Good (precise column matching)

**Tier 3 (Keyword Matching)**:
- Keywords: `["comprehensive", "territory", "zip", "disruption", "motorcycle", "mcy", "rate"]`
- Found: 0 additional (already had 27)
- Quality: Fair (broad matching)

**Final**: 27 tables retrieved (8 semantic + 19 column search)

## Without Keywords - What Would Happen?

**Scenario**: Only using semantic search

```
Query: "hurricane premium calculation"
Semantic search: Finds 1 table
Result: Only 1 table retrieved
Problem: Missing 26 potentially relevant tables
Impact: Lower accuracy, missing data
```

**With Keywords**:
```
Query: "hurricane premium calculation"
Semantic search: Finds 1 table
Column search: Finds 26 additional tables
Result: 27 tables retrieved
Impact: Comprehensive coverage, better accuracy
```

## Keyword Extraction Logic

### Column Keywords (Tier 2)
More specific, technical terms:

```python
if "base rate" in query:
    keywords = ["base_rate", "ho3_a_rate", "coverage_a"]
    
if "territory" in query:
    keywords = ["territory", "zip_code"]
    
if "grg" in query:
    keywords = ["grg", "collision_rating_group"]
```

### Table Name Keywords (Tier 3)
More general, document-level terms:

```python
if "hurricane" in query:
    keywords = ["hurricane"]
    
if "ho3" in query:
    keywords = ["ho3", "homeowner", "maps"]
    
if "rate" in query:
    keywords = ["rate"]
```

## Performance Impact

**Without Keywords**:
- Tables retrieved: 1-8
- Coverage: ~0.6-0.7% of 1,227 tables
- Accuracy: Lower (missing relevant tables)

**With Keywords**:
- Tables retrieved: 27
- Coverage: ~2.2% of 1,227 tables
- Accuracy: Higher (comprehensive coverage)
- Processing time: +2-5 seconds (column search overhead)

## Conclusion

**Keywords are essential because:**
1. **Scale**: 1,227 tables is too many to search semantically alone
2. **Coverage**: Semantic finds 1-8 tables, keywords add 19-26 more
3. **Precision**: Column search finds exact technical term matches
4. **Reliability**: Fallback ensures we always find relevant tables
5. **Complementary**: Each tier finds different types of tables

**The three-tier system is a practical solution** for finding relevant tables in a large database (1,227 tables) where:
- Semantic search provides quality but limited quantity
- Column search provides precision for technical terms
- Keyword matching provides broad coverage as fallback

Without keywords, the system would be significantly less effective, retrieving only 1-8 tables instead of 27, missing 70-96% of potentially relevant data.

