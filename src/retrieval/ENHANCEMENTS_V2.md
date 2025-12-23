# RAG Agent Enhancements V2

## Overview

Enhanced the RAG agent to better handle list questions and calculation questions through logic-based improvements rather than hardcoded instructions.

## Changes Made

### 1. Enhanced Prompt Template (prompt_template_v2.txt)

#### Added Calculation Instructions
- New section: "For calculation questions" with logic-based guidance
- Instructs LLM to:
  - Identify required inputs from the question
  - Search ALL tables systematically for values
  - Look for rate tables, factor tables, calculation tables
  - Match rows based on question parameters
  - Combine information from multiple tables
  - Show work: values found → calculation → result

#### Improved List Instructions
- Enhanced list query instruction to look for:
  - Table of contents
  - Index pages
  - Summary tables
  - Numbered items in sequence

### 2. Calculation Query Detection

**New Method: `_is_calculation_query()`**
- Detects calculation questions using keywords:
  - `calculate`, `compute`, `determine`, `find`, `what is`
  - `premium`, `rate`, `factor`, `multiply`, `multiplier`
  - `deductible`, `discount`, `surcharge`, `adjustment`

**New Method: `_extract_calculation_inputs()`**
- Extracts inputs needed for calculation:
  - Policy types (HO3, HO-3, etc.)
  - Coverage amounts ($750,000, etc.)
  - Distances (3000 feet, etc.)
  - Calculation terms (base rate, deductible factor, etc.)
  - Location terms (coast, coastline, etc.)

### 3. Enhanced Retrieval for List Questions

**Improved List Query Handling:**
- **Strategy 1**: Search for table of contents/index pages
  - Queries: "table of contents", "index", "summary", "list of rules", "all rules"
  - Retrieves up to 10 TOC/index results
  
- **Strategy 2**: Enhanced term extraction
  - Analyzes top 15 results (increased from 10)
  - Extracts numbered items (e.g., "Rule C-1", "Factor 1")
  - Extracts capitalized phrases
  - Searches up to 20 terms (increased from 15)
  - Retrieves top 2 results per term (increased from 1)

### 4. Enhanced Retrieval for Calculation Questions

**Priority 1.5: Calculation-Specific Table Search**
- When calculation inputs are detected:
  - Searches for tables with calculation-related keywords
  - Looks for rate tables, factor tables, multiplier tables
  - Adds calculation inputs to specific value search
  - Prioritizes tables containing base rates, factors, etc.

### 5. Dynamic Prompt Instructions

**List Instruction:**
- Only added for "list all" queries
- Guides LLM to be comprehensive
- Mentions table of contents and numbered sequences

**Calculation Instruction:**
- Only added for calculation queries
- Guides LLM on how to find and use calculation inputs
- Emphasizes systematic table searching

## Benefits

1. **Flexible**: Logic-based detection, not hardcoded for specific questions
2. **Comprehensive**: Better retrieval for list questions (TOC, index, numbered items)
3. **Systematic**: Calculation questions now guide LLM to search for required inputs
4. **Generic**: Works for any insurance filing question type

## Testing Recommendations

1. **List Questions**: Test with various "list all" queries to verify comprehensive retrieval
2. **Calculation Questions**: Test with different calculation types (premiums, rates, factors)
3. **Mixed Questions**: Test questions that combine list and calculation elements
4. **New Question Types**: Verify the system adapts to new question patterns

## Code Changes Summary

- `rag_agent.py`: Added `_is_calculation_query()`, `_extract_calculation_inputs()`, enhanced list query retrieval
- `retrieval_logic.py`: Added `calculation_inputs` parameter to `full_table_retrieval()`
- `prompt_template_v2.txt`: Added calculation instructions section and `{calculation_instruction}` placeholder

