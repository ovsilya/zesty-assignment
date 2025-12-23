# Prompt Template

The LLM prompt is now stored in `prompt_template.txt` instead of being hardcoded in the Python code.

## Template File

**Location**: `src/retrieval/prompt_template.txt`

## Template Variables

The template uses Python string formatting with the following placeholders:

- `{list_instruction}` - Conditional instruction for "list all" queries (empty string for regular queries)
- `{context_block}` - The full context including semantic text chunks and table data
- `{question}` - The user's question

## Modifying the Prompt

To modify the prompt:

1. Edit `src/retrieval/prompt_template.txt`
2. Use the placeholders `{list_instruction}`, `{context_block}`, and `{question}` where dynamic content should be inserted
3. The changes will take effect immediately (no code changes needed)

## Error Handling

If the template file is not found, the system will raise a `FileNotFoundError` with a clear message. The template file is required for the RAG agent to function.

## Example Template Structure

```
You are a careful RAG assistant...

{list_instruction}

Context:
{context_block}

Instructions:
- Rule 1
- Rule 2
...

Question: {question}

Answer:
```

