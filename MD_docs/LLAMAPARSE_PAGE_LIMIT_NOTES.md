# LlamaParse Page Limit Issue

## Problem
LlamaParse appears to be processing only the first page of multi-page PDFs, even though the PDFs have multiple pages (e.g., 5 pages, but only ~16 lines of output).

## Findings

1. **PDF Page Counts:**
   - `(211237071-177742463)-MCY_Exhibits.pdf`: 5 pages
   - Output: Only 16 lines (likely first page only)

2. **LlamaParse Free Tier:**
   - Free tier: 1,000 pages per day
   - **May have a limit of 1 page per document** on free tier
   - This would explain why only first page is processed

3. **Possible Solutions:**

   a. **Check LlamaParse Account Settings:**
      - Log into https://cloud.llamaindex.ai
      - Check if there's a "pages per document" limit on free tier
      - May need to upgrade to process full documents

   b. **Use Page Parameters (if supported):**
      ```python
      parser = LlamaParse(
          api_key=api_key,
          result_type="markdown",
          max_pages=None,  # Process all pages
          target_pages=None,  # Process all pages
      )
      ```
      Note: These parameters may not be available in all versions.

   c. **Use Direct API Methods:**
      LlamaParse has methods like `aget_tables()` and `aget_json()` that might provide better access to structured data.

   d. **Process Pages Individually:**
      If free tier limits to 1 page, you could:
      - Split PDF into individual pages
      - Process each page separately
      - Combine results

## Current Status

The test script (`scripts/test_llamaparse_tables.py`) now:
- Checks PDF page count
- Warns if output seems truncated
- Attempts to use direct API methods if available

## Recommendation

1. **Check your LlamaParse account** to confirm free tier limits
2. **Test with a paid tier** if available to see if full document processing works
3. **Consider hybrid approach**: Use LlamaParse for complex tables on first page, Unstructured for remaining pages
4. **Alternative**: Process documents page-by-page if free tier allows multiple 1-page documents

## Next Steps

1. Verify LlamaParse account settings
2. Test with a single-page PDF to confirm it works
3. If free tier is limited, consider:
   - Upgrading to paid tier
   - Using Unstructured for full documents
   - Hybrid approach (LlamaParse for key pages, Unstructured for rest)

