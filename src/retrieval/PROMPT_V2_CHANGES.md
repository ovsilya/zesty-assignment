# Prompt Template V2 - Generic Version

## Overview

The v2 prompt template removes all hardcoded examples that were specific to the test questions (EF_1, EF_2, EF_3) and replaces them with generic patterns that work for any insurance filing question.

## Key Changes

### 1. Removed Hardcoded Values

**Before (v1):**
- `'$604' as 604` - Specific answer from EF_3
- `'Comprehensive coverage'` - Specific coverage type from EF_2
- `Territory 117 includes ZIP codes 89030, 89110, 89115` - Specific territory/ZIP codes from EF_2
- `Territory 117... East of Nellis Boulevard area, NOT Territory 9` - Very specific example from EF_2

**After (v2):**
- `'$500.00' as 500.00` - Generic monetary example
- `'specific type of coverage, factor, or attribute'` - Generic coverage reference
- `Territory A is defined as including 'portions of ZIP Code X not in Territory B'` - Generic pattern
- Uses placeholder patterns like `[area]`, `[other territory]`, `Territory A`, `Territory B`

### 2. Expanded Examples

**Before (v1):**
- `motorcycle models, territory numbers, zip codes` - Limited to EF_2 context

**After (v2):**
- `vehicle models, territory numbers, zip codes, policy numbers, claim IDs` - Broader scope

### 3. More Generic Geographic Patterns

**Before (v1):**
- `'West of X Street'` - Only one direction
- `'East of X Street'` - Specific to EF_2 pattern

**After (v2):**
- `'North of X Street', 'South of Y Avenue'` - Multiple directions
- `'East of X Street'` and `'West of X Street'` - Generic pattern without specific values

### 4. Improved Clarity

- Changed "motorcycle models" to "vehicle models" (more general)
- Changed "rate change value" to "value or rate" (broader)
- Changed "territory numbers" to "territory numbers or codes" (more flexible)
- Added "policy numbers, claim IDs" to searchable items

## How to Use V2

### Option 1: Replace Current Template

```bash
# Backup current template
cp src/retrieval/prompt_template.txt src/retrieval/prompt_template.txt.backup

# Use v2 as the active template
cp src/retrieval/prompt_template_v2.txt src/retrieval/prompt_template.txt
```

### Option 2: Update Code to Use V2

Modify `rag_agent.py` to load v2:

```python
# In __init__ method
prompt_template_path = Path(__file__).parent / "prompt_template_v2.txt"
```

### Option 3: A/B Testing

Keep both versions and switch between them for testing:

```python
# In __init__ method
use_v2 = os.getenv('USE_PROMPT_V2', 'false').lower() == 'true'
template_name = "prompt_template_v2.txt" if use_v2 else "prompt_template.txt"
prompt_template_path = Path(__file__).parent / template_name
```

## Benefits of V2

1. **No Hardcoded Examples**: Works for any insurance filing question, not just the 3 test questions
2. **More Flexible**: Generic patterns apply to various scenarios
3. **Broader Scope**: Handles more types of questions (policies, claims, etc.)
4. **Future-Proof**: Won't need updates when new question types are added

## Testing Recommendations

1. Test with the original 3 questions to ensure accuracy is maintained
2. Test with new questions to verify flexibility
3. Compare v1 vs v2 performance on a diverse set of questions
4. Monitor for any degradation in specific question types

## Migration Notes

- The logic and structure remain the same
- Only examples and specific values were changed
- All CRITICAL and IMPORTANT instructions are preserved
- Template variables (`{list_instruction}`, `{context_block}`, `{question}`) unchanged

