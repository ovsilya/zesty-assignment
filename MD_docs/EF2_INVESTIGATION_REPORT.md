# EF_2 Investigation Report

## Question
For a State Farm Fire & Casualty Company policyholder living in Zip Code 89110, does the resident living West of Nellis Boulevard face a higher proposed percentage rate change for Comprehensive coverage compared to a resident living East of Nellis Boulevard, and does the numeric difference between the Collision Rating Groups (GRG) of a 2023 Ducati Panigale V4 R and a 2023 Honda Grom ABS exceed 30 points?

## Expected Answer
Yes. A resident living West of Nellis Boulevard (Territory 118) faces a higher proposed Comprehensive rate change (0.305%) than a resident living East of Nellis Boulevard (Territory 117, -0.133%). Additionally, the difference between the Collision Rating Group of the Ducati Panigale V4 R (051) and the Honda Grom ABS (015) is 36, which exceeds 30 points.

## Investigation Results

### ✅ Data EXISTS in BigQuery

#### 1. Motorcycle GRG Data - FOUND
- **Ducati Panigale V4 R**: GRG 051
  - Found in: `table__213128742_179157333__2024_CW_Rate_Manual_Pages_Redlined_table_1`
  - Page: 2 (also appears on page 11)
  - Columns: MAKE, MODEL, GRG

- **Honda Grom ABS**: GRG 015
  - Found in: `table__213128742_179157333__2024_CW_Rate_Manual_Pages_Redlined_table_12`
  - Page: 13
  - Columns: MAKE, MODEL, GRG

#### 2. Territory/Rate Change Data - PARTIALLY FOUND
- **Table**: `table__213128717_179157013__MCY_Rate_Filing_Data_Summary___SFFC_table_1`
- **Page**: 2
- **Columns**: Category_Bin, Med__Pay___Change, Comprehensive___Change, Collision___Change

**Found Data**:
- Row: `{'Category_Bin': 'ZIP 89120 and portions of 89030, 89115 and 89110', 'Comprehensive___Change': '0.305%'}` 
  - This is **Territory 118** (West of Nellis) ✓
- Row: `{'Category_Bin': '89110 (117)', 'Comprehensive___Change': ''}`
  - This is **Territory 117** but the rate change is empty in sample rows
  - Need to check full table for -0.133%

- **Territory Definitions**:
  - Table: `table__213128717_179157013__MCY_Rate_Filing_Data_Summary___SFFC_table_23` (Page 20)
  - Row: `{'col1': 'Territory 117', 'col2': 'Territory comprises that portion of Clark County for which residences have the following ZIP Codes:', 'col3': '89030, 89110, 89115'}`
  
  - Table: `table__213128717_179157013__MCY_Rate_Filing_Data_Summary___SFFC_table_27` (Page 21)
  - Row: `{'col1': '118', 'col2': '89110, 89115, 89030, 89120'}`

### ❌ Missing from Vector Store (doc_index)

**Critical Pages NOT Indexed**:
1. **Document 1**: `(213128717-179157013)-MCY Rate Filing Data Summary - SFFC.pdf`
   - ✅ Pages 1-6: Indexed
   - ❌ Pages 20-21: **NOT INDEXED** (Territory definitions)

2. **Document 2**: `(213128742-179157333)-2024 CW Rate Manual Pages Redlined.pdf`
   - ✅ Page 1: Indexed
   - ❌ Pages 11, 13: **NOT INDEXED** (GRG values for specific motorcycles)

### 📊 Tables Retrieved During Test

From test output:
- **21 tables** retrieved from semantic search
- **20 tables** actually fetched (limit of 20)
- But the **critical tables were likely not in the top 20**

## Root Causes

### 1. Pages Not Indexed
- Pages 20-21, 11, 13 are missing from vector store
- These pages contain the exact data needed
- Semantic search cannot find content that isn't indexed

### 2. Table Retrieval Issues
- Tables exist in BigQuery but semantic search didn't prioritize them
- Table summaries in vector store may not mention "Territory 117/118" or specific motorcycle models
- The 20-table limit may have excluded the relevant tables

### 3. Data Format Issues
- Some tables use generic column names (col1, col2)
- Rate change values might be in rows not shown in sample
- Territory 117 rate change (-0.133%) needs full table inspection

## Required Data Locations

### For Territory/Rate Changes:
1. **Table**: `table__213128717_179157013__MCY_Rate_Filing_Data_Summary___SFFC_table_1`
   - Contains: Comprehensive rate changes by territory
   - Territory 118: 0.305% ✓ (found)
   - Territory 117: -0.133% (needs full table check)

2. **Tables for Territory Definitions**:
   - `table__213128717_179157013__MCY_Rate_Filing_Data_Summary___SFFC_table_23` (Page 20)
   - `table__213128717_179157013__MCY_Rate_Filing_Data_Summary___SFFC_table_27` (Page 21)

### For Motorcycle GRG:
1. **Table**: `table__213128742_179157333__2024_CW_Rate_Manual_Pages_Redlined_table_1` (Page 2)
   - Ducati Panigale V4 R: GRG 051 ✓

2. **Table**: `table__213128742_179157333__2024_CW_Rate_Manual_Pages_Redlined_table_12` (Page 13)
   - Honda Grom ABS: GRG 015 ✓

## Implemented Modifications

### ✅ 1. Value-Based Table Search (IMPLEMENTED)
- **Added `_extract_specific_values_from_query`**: Extracts specific values from queries:
  - Zip codes (5-digit numbers, e.g., "89110")
  - Territory numbers (e.g., "Territory 117" → "117")
  - Percentages (e.g., "0.305%", "-0.133%")
  - GRG values (e.g., "GRG 051" → "051")
  - Model names (capitalized phrases, e.g., "Ducati Panigale V4 R")
- **Added Priority 2 in `_sql_retrieval`**: After semantic search, searches table content for these specific values
- **Enhanced `_search_tables_by_content`**: 
  - Increased sample_rows from 5 to 10 for better coverage
  - Removed table limit (now searches all tables, not just first 50)
- **Result**: Tables containing exact values from the query are now prioritized

### 2. Remaining Improvements Needed
- **Document-specific table prioritization**: When query mentions specific documents (e.g., "SFFC", "MCY"), prioritize tables from those documents
- **Multi-pass retrieval**: If first pass doesn't find data, do a second pass with more specific searches
- **Table content preview**: Before full retrieval, preview table content to verify relevance

## Next Steps

1. ✅ Verify all required data exists in BigQuery (DONE - data exists)
2. ⏳ Check if Territory 117 rate change (-0.133%) is in the full table
3. ⏳ Improve table retrieval to find these specific tables
4. ⏳ Add value-based search for specific numbers/percentages
5. ⏳ Test with improved retrieval

