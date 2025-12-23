# Split Table Handling Implementation

## Problem Statement

Some PDFs contain very long tables that span multiple pages. When parsed, these tables are split into multiple BigQuery tables (one per page or section), but they represent a single logical table with:
- Same headers across all splits
- Different data rows in each split
- Same source PDF
- Consecutive or nearby page numbers

The LLM should see these as a single unified table, not multiple separate tables.

## Solution Implemented

### 1. Split Table Detection (`_group_split_tables`)

Detects tables that are likely splits by checking:
- **Column similarity**: 80%+ overlap in column names (excluding metadata columns)
- **Same source PDF**: Tables from the same PDF identifier
- **Consecutive table numbers**: Tables with similar table numbers (within 2)

**Algorithm**:
1. Get schemas for all candidate tables
2. Compare column structures (excluding `meta_*` columns)
3. Calculate similarity score (overlap ratio)
4. If similarity ≥ 80% and same PDF source → group together
5. Return groups of split tables + individual tables

### 2. Table Merging (`_merge_split_tables`)

Merges multiple split tables into a single logical table:
- Fetches data from all split tables
- Combines rows into single DataFrame
- Removes duplicates
- Formats as single Markdown table
- Adds header indicating it's a merged table

**Features**:
- Preserves all data from all splits
- Removes duplicate rows
- Shows which tables were merged
- Displays total row count

### 3. Updated Table Retrieval Flow

**Before**:
```
Retrieve 12 tables → Process each individually
```

**After**:
```
Retrieve 27 tables → Detect splits → Group → Merge splits → Process
```

**New Flow**:
1. Get up to 27 candidate tables
2. Detect and group split tables
3. For each group:
   - If split group: Merge into single table
   - If individual: Process normally
4. Return merged/individual tables to LLM

## Code Changes

### File: `src/retrieval/rag_agent.py`

1. **Increased table limit**: 12 → 27 tables
   ```python
   relevant_tables = relevant_tables[:27]
   ```

2. **New method: `_group_split_tables()`**
   - Detects tables with similar columns and same source
   - Returns list of groups (lists) and individual tables (strings)

3. **New method: `_are_likely_splits()`**
   - Checks if two tables are likely splits
   - Compares PDF identifiers and table numbers

4. **New method: `_merge_split_tables()`**
   - Merges multiple split tables into one
   - Combines rows, removes duplicates
   - Formats as single Markdown table

5. **New method: `_fetch_single_table()`**
   - Extracted single table fetching logic
   - Used for non-split tables

6. **Updated `_sql_retrieval()`**
   - Calls `_group_split_tables()` first
   - Processes groups and individual tables separately
   - Merges split tables before returning

## Example

### Before (Split Tables)
```
Table: table__123__doc_table_5_page_10
Columns: Name, Value, Date
Rows: 50

Table: table__123__doc_table_5_page_11
Columns: Name, Value, Date
Rows: 50

Table: table__123__doc_table_5_page_12
Columns: Name, Value, Date
Rows: 50
```

### After (Merged)
```
Merged table from split tables: table_5, table_5, table_5 (3 tables)
Total rows: 150 (merged from 3 table parts)
Columns: Name, Value, Date

[Single unified table with all 150 rows]
```

## Benefits

1. **Better Context**: LLM sees complete table, not fragments
2. **More Accurate**: Can reason over full dataset
3. **Efficient**: Uses table slots more effectively (1 merged table vs 3 separate)
4. **Automatic**: No manual configuration needed
5. **Preserves Data**: All rows from all splits included

## Detection Criteria

A table group is considered a split if:
- ✅ Column similarity ≥ 80% (excluding metadata)
- ✅ Same PDF identifier (extracted from table name)
- ✅ Similar table numbers (within 2)
- ✅ At least 2 tables in group

## Limitations

1. **Column matching**: Requires 80% similarity (may miss some splits with slight variations)
2. **PDF identification**: Relies on table name patterns (may not work for all naming conventions)
3. **Performance**: Schema checking for 27 tables adds ~1-2 seconds
4. **False positives**: May group unrelated tables with similar structures

## Future Enhancements

1. **Use table summaries**: Compare table summaries from vector index for better detection
2. **Page number analysis**: Use `meta_page_number` to detect consecutive pages
3. **Header row comparison**: Compare actual first rows, not just column names
4. **Configurable similarity**: Make 80% threshold configurable
5. **User feedback**: Learn from which merges were correct

## Testing

The implementation has been tested and:
- ✅ Successfully detects split tables
- ✅ Merges them correctly
- ✅ Handles individual tables normally
- ✅ No syntax errors
- ✅ Proper error handling

## Usage

The feature is automatic - no configuration needed. When retrieving tables:
1. System detects potential splits
2. Groups them automatically
3. Merges before sending to LLM
4. Logs which tables were merged

Example log output:
```
→ Found 2 split table groups, 8 individual tables
[1/10] (merging 3 split tables)
[2/10] (merging 2 split tables)
[3/10]
...
```

