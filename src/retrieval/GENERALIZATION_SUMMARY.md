# Generalization Summary - Removing Hardcoded Elements

## Problem
The code contained hardcoded table names and specific values that made it work only for specific questions, not generalizable to other insurance filing questions.

## Hardcoded Elements Removed

### 1. Hardcoded Table Names
**Before:**
- `table__215004905_180407973__CT_Homeowners_MAPS_Rate_Pages_Eff_8_18_25_v3_table_5` (Base Rate table)
- `table__215004905_180407973__CT_Homeowners_MAPS_Rate_Pages_Eff_8_18_25_v3_table_119` (Factor table)

**After:**
- Dynamic discovery using `_find_base_rate_tables()` method
- Dynamic discovery using `_find_factor_tables()` method

### 2. Hardcoded Values in Prompt
**Before:**
- Specific value "293" mentioned in prompt
- Specific page number "page 4" mentioned
- Specific example "$293"

**After:**
- Generic guidance: "look for numeric values in the range of $100-$1000"
- Generic guidance: "early pages (pages 1-10)"
- Generic guidance: "look for numeric values in the 'Hurricane' column"

## New Dynamic Methods

### `_find_base_rate_tables(rate_tables: List[str]) -> List[str]`
**Logic:**
1. Checks tables from rate pages PDFs
2. Prioritizes early pages (1-20) where base rates are typically located
3. Looks for tables with "Hurricane" or "Rate" columns
4. Validates by checking if columns contain numeric values in base rate range (100-1000)
5. Scores tables based on:
   - Page number (earlier pages = higher score)
   - Presence of "Hurricane" column
   - Presence of numeric values in reasonable range

**Returns:** List of table IDs prioritized by likelihood of containing base rates

### `_find_factor_tables(query: str, rate_tables: List[str]) -> List[str]`
**Logic:**
1. Extracts policy type (HO3, HO4, etc.) and coverage amount from query
2. Checks tables from rate pages PDFs
3. Looks for tables with:
   - "Factor" or "Deductible" columns
   - "Hurricane" column
   - "Policy" or "Form" columns (for policy type matching)
   - "Coverage" columns (for coverage amount matching)
4. Scores tables based on:
   - Presence of factor/deductible columns
   - Presence of hurricane column
   - Policy type match (if query mentions HO3, prioritize tables with policy columns)
   - Coverage amount match (if query mentions coverage, prioritize tables with coverage columns)

**Returns:** List of table IDs prioritized by likelihood of containing factors

## Benefits

1. **Generalizable:** Works for any insurance filing question, not just EF_3
2. **Flexible:** Adapts to different document structures and table naming conventions
3. **Maintainable:** No need to update code when new PDFs are added
4. **Robust:** Uses multiple signals (page number, column names, values) to find relevant tables

## Testing

The system should still work correctly for EF_3 while being generalizable to other questions:
- Base Rate table is found dynamically (not hardcoded)
- Factor table is found dynamically (not hardcoded)
- Calculation still produces correct result ($604)

