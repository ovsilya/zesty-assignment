from __future__ import annotations

"""
PDF parsing utilities using LlamaParse only.

Responsibilities:
- Extract text, tables, and charts from PDFs using LlamaParse JSON
- Return a lightweight Python structure suitable for downstream indexing
"""

import os
import asyncio
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


try:
    from llama_parse import LlamaParse
except Exception:  # pragma: no cover - optional dependency
    LlamaParse = None  # type: ignore

try:
    # Unstructured is used for robust text extraction (DOC index)
    from unstructured.partition.pdf import partition_pdf
except Exception:  # pragma: no cover - optional dependency
    partition_pdf = None  # type: ignore


@dataclass
class ParsedElement:
    """Normalized representation of a parsed PDF text element."""

    type: str
    text: str
    page_number: Optional[int]
    metadata: Dict[str, Any]


@dataclass
class ParsedTable:
    """Representation of an extracted table.
    
    Uses structured data from LlamaParse JSON:
    - rows: List of lists (preferred)
    - csv: CSV string (alternative)
    """

    rows: Optional[List[List[Any]]] = None  # For LlamaParse JSON (preferred)
    csv: Optional[str] = None  # Alternative structured format
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Ensure at least one format is provided."""
        if self.rows is None and self.csv is None:
            raise ValueError("ParsedTable must have at least one of: rows or csv")
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ParsedChart:
    """Representation of an extracted chart.
    
    Charts from LlamaParse can contain:
    - data: Structured data points
    - metadata: Chart type, title, axes labels
    - image: Chart image data (if available)
    """

    data: Optional[Dict[str, Any]] = None
    chart_type: Optional[str] = None
    title: Optional[str] = None
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ParsedDocument:
    """Parsed representation of a single PDF."""

    pdf_path: str
    text_elements: List[ParsedElement]
    tables: List[ParsedTable]
    charts: List[ParsedChart]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pdf_path": self.pdf_path,
            "text_elements": [asdict(e) for e in self.text_elements],
            "tables": [asdict(t) for t in self.tables],
            "charts": [asdict(c) for c in self.charts],
        }


class PDFParser:
    """Unified PDF parser using LlamaParse only for text, tables, and charts."""

    def __init__(self) -> None:
        if LlamaParse is None:
            raise ImportError(
                "llama-parse is not installed. Please install `llama-parse`."
            )
        
        api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY environment variable not set")
        
        print("  → Initializing LlamaParse (unified parser for text, tables, and charts)...")
        # Note: We'll create parser instances per PDF to avoid event loop issues
        self.api_key = api_key
        print("  ✓ LlamaParse ready")

    def _get_pages_list(self, json_data: Any) -> Optional[List[Dict[str, Any]]]:
        """Extract pages list from LlamaParse JSON response."""
        pages_list = None
        
        # Case 1: It's a list directly (most common from aget_json)
        if isinstance(json_data, list) and len(json_data) > 0:
            first_elem = json_data[0]
            if isinstance(first_elem, dict) and 'pages' in first_elem:
                pages_list = first_elem['pages']
        
        # Case 2: It's a dict with 'data' key
        elif isinstance(json_data, dict) and 'data' in json_data:
            data_value = json_data['data']
            
            # data might be a JSON string or already parsed
            if isinstance(data_value, str):
                try:
                    import json as json_module
                    data_value = json_module.loads(data_value)
                except:
                    try:
                        import ast
                        data_value = ast.literal_eval(data_value)
                    except:
                        pass
            
            # Now data_value should be a list
            if isinstance(data_value, list) and len(data_value) > 0:
                first_elem = data_value[0]
                if isinstance(first_elem, dict) and 'pages' in first_elem:
                    pages_list = first_elem['pages']
        
        return pages_list

    def _extract_text_from_json(self, json_data: Any, pdf_path: Path) -> List[ParsedElement]:
        """Extract text elements from LlamaParse JSON output.
        
        Prefer page-level markdown ("md") over raw "text" to avoid chart visuals.
        - Use page["md"] or page["markdown"] when available.
        - Strip out HTML/markdown tables (tables come from "rows" of items[type="table"]).
        """
        text_elements: List[ParsedElement] = []
        pages_list = self._get_pages_list(json_data)
        
        if not pages_list:
            return text_elements
        
        import re
        
        for page_data in pages_list:
            page_num = page_data.get('page')
            page_md = page_data.get('md') or page_data.get('markdown')
            
            if page_md and isinstance(page_md, str) and page_md.strip():
                # Remove HTML table blocks
                text_clean = re.sub(r'<table>.*?</table>', '', page_md, flags=re.DOTALL | re.IGNORECASE)
                # Remove markdown table blocks (lines starting with '|' that form tables)
                text_clean = re.sub(r'^\|.*\|.*$', '', text_clean, flags=re.MULTILINE)
                
                if text_clean.strip():
                    # Split into paragraphs by blank lines
                    paragraphs = [p.strip() for p in text_clean.split('\n\n') if p.strip() and len(p.strip()) > 20]
                    for para in paragraphs:
                        # Skip leftover table-like lines just in case
                        if para.strip().startswith('|') and '|' in para[1:]:
                            continue
                        
                        text_elements.append(
                            ParsedElement(
                                type='paragraph',
                                text=para,
                                page_number=page_num,
                                metadata={
                                    "source": str(pdf_path),
                                    "parser": "llamaparse",
                                    "extracted_from": "page_md",
                                },
                            )
                        )
            else:
                # Fallback: use "text" only if no markdown is available
                page_text = page_data.get('text')
                if page_text and isinstance(page_text, str) and page_text.strip():
                    text_clean = page_text.strip()
                    paragraphs = [p.strip() for p in text_clean.split('\n\n') if p.strip() and len(p.strip()) > 20]
                    for para in paragraphs:
                        if para.strip().startswith('|') and '|' in para[1:]:
                            continue
                        text_elements.append(
                            ParsedElement(
                                type='paragraph',
                                text=para,
                                page_number=page_num,
                                metadata={
                                    "source": str(pdf_path),
                                    "parser": "llamaparse",
                                    "extracted_from": "page_text_fallback",
                                },
                            )
                        )
        
        return text_elements

    def _extract_text_with_unstructured(self, pdf_path: Path) -> List[ParsedElement]:
        """Extract text portions using Unstructured (for DOC index).
        
        - Uses high-resolution strategy with table inference.
        - Skips elements categorized as tables (tables handled by LlamaParse).
        """
        text_elements: List[ParsedElement] = []

        if partition_pdf is None:
            print("    ⚠ Unstructured (partition_pdf) not available, skipping Unstructured text extraction.")
            return text_elements

        try:
            print("    → Extracting text with Unstructured (partition_pdf)...")
            elements = partition_pdf(
                filename=str(pdf_path),
                strategy="hi_res",
                infer_table_structure=True,
                extract_images_in_pdf=False,
            )

            for el in elements:
                # Category: e.g., "Title", "NarrativeText", "Table", etc.
                category = getattr(el, "category", "") or ""
                category_lower = str(category).lower()

                # Skip tables from Unstructured - tables are handled via LlamaParse
                if category_lower == "table":
                    continue

                # Get raw text
                text = str(el).strip()
                if not text:
                    continue

                # Page number if available
                page_number = None
                meta = getattr(el, "metadata", None)
                if meta is not None and hasattr(meta, "page_number"):
                    page_number = meta.page_number

                text_elements.append(
                    ParsedElement(
                        type=category_lower or "paragraph",
                        text=text,
                        page_number=page_number,
                        metadata={
                            "source": str(pdf_path),
                            "parser": "unstructured",
                        },
                    )
                )

            print(f"    ✓ Extracted {len(text_elements)} text elements with Unstructured")
        except Exception as e:
            print(f"    ⚠ Unstructured text extraction error: {e}")

        return text_elements

    def _extract_table_data(self, table_elem: Dict[str, Any]) -> tuple:
        """Extract rows from a LlamaParse table element.
        
        Uses only the \"rows\" key from items with type=\"table\" (as per sample_extracted.json).
        We intentionally ignore other representations like \"csv\", \"md\", or \"html\" to
        keep the logic simple and deterministic.
        
        Returns:
            (rows, None): Tuple of rows (list of lists) and None for csv.
        """
        rows = None
        
        if "rows" in table_elem:
            rows = table_elem["rows"]
            if rows and isinstance(rows, list) and len(rows) > 0:
                # Validate rows structure
                if not all(isinstance(row, list) for row in rows):
                    rows = None
        
        return rows, None

    def _extract_tables_from_json(self, json_data: Any, pdf_path: Path) -> List[ParsedTable]:
        """Extract tables from LlamaParse JSON output.
        
        Uses items array with type="table" and extracts "rows" (as per sample_extracted.json).
        """
        tables: List[ParsedTable] = []
        pages_list = self._get_pages_list(json_data)
        
        if pages_list:
            for page_data in pages_list:
                page_num = page_data.get('page')
                items = page_data.get('items', [])
                
                for item in items:
                    if item.get('type') == 'table':
                        is_perfect = item.get('isPerfectTable', False)
                        has_rows = 'rows' in item and item['rows']
                        
                        print(
                            f"      → Found table on page {page_num} "
                            f"(perfect={is_perfect}, has_rows={bool(has_rows)})...",
                            end=" ",
                            flush=True,
                        )
                        
                        # Extract rows only (simple, deterministic behavior)
                        rows, _ = self._extract_table_data(item)
                        
                        if not rows:
                            print("⚠ (no rows data, skipping)")
                            continue
                        
                        row_count = len(rows)
                        col_count = len(rows[0]) if rows and len(rows[0]) > 0 else 0
                        print(f"✓ {row_count} rows × {col_count} cols")
                        
                        tables.append(
                            ParsedTable(
                                rows=rows,
                                csv=None,
                                page_number=page_num,
                                metadata={
                                    "source": str(pdf_path),
                                    "parser": "llamaparse",
                                    "isPerfectTable": is_perfect,
                                },
                            )
                        )
        
        return tables

    def _extract_charts_from_json(self, json_data: Any, pdf_path: Path) -> List[ParsedChart]:
        """Extract charts from LlamaParse JSON output."""
        charts: List[ParsedChart] = []
        pages_list = self._get_pages_list(json_data)
        
        if pages_list:
            for page_data in pages_list:
                page_num = page_data.get('page')
                items = page_data.get('items', [])
                
                for item in items:
                    item_type = item.get('type', '').lower()
                    
                    # Check for chart types
                    if item_type in ['chart', 'figure', 'graph', 'plot', 'visualization']:
                        print(f"      → Found chart on page {page_num}...", end=" ", flush=True)
                        
                        # Extract chart data
                        chart_data = item.get('data') or item.get('chart_data') or {}
                        chart_type = item.get('chart_type') or item.get('type', 'chart')
                        title = item.get('title') or item.get('caption') or item.get('text', '')
                        
                        charts.append(
                            ParsedChart(
                                data=chart_data if isinstance(chart_data, dict) else {'raw': chart_data},
                                chart_type=chart_type,
                                title=title[:200] if title else None,  # Limit title length
                                page_number=page_num,
                                metadata={
                                    "source": str(pdf_path),
                                    "parser": "llamaparse",
                                    "raw_item": {k: v for k, v in item.items() if k not in ['data', 'chart_data']},
                                },
                            )
                        )
                        print(f"✓ ({chart_type})")
        
        return charts

    def parse_pdf(self, pdf_path: Path) -> ParsedDocument:
        """Parse a single PDF file.
        
        Responsibilities:
        - Use LlamaParse JSON for structured data (tables, charts)
        - Use Unstructured for text extraction (DOC index)
        """
        text_elements: List[ParsedElement] = []
        tables: List[ParsedTable] = []
        charts: List[ParsedChart] = []
        
        print(f"    → Extracting content with LlamaParse...")
        
        try:
            # Create parser for JSON (structured data: tables, charts, text)
            parser_json = LlamaParse(
                api_key=self.api_key,
                result_type="json",
                verbose=False,
                num_workers=1,
                extract_charts=True,
                system_prompt="Convert visual bar charts to structured tables with categories/bins as rows and values as columns. Check for the axis and legends.",  # Guide for table-like output
                auto_mode=True,
                outlined_table_extraction=True,
                adaptive_long_table=True,
            )
            
            # Step 1: Get JSON (contains pages with items array, tables, charts)
            async def get_json():
                try:
                    result = await parser_json.aget_json(str(pdf_path))
                    return result
                except Exception as e:
                    print(f"      ⚠ aget_json error: {e}")
                    return None
            
            json_data = asyncio.run(get_json())
            
            if json_data:
                # Handle different return types from aget_json
                if isinstance(json_data, dict) and 'data' in json_data:
                    data_value = json_data['data']
                    if isinstance(data_value, str):
                        try:
                            import json as json_module
                            data_value = json_module.loads(data_value)
                        except:
                            try:
                                import ast
                                data_value = ast.literal_eval(data_value)
                            except:
                                pass
                    if isinstance(data_value, list):
                        json_data = data_value
                
                # Extract structured data from JSON
                tables = self._extract_tables_from_json(json_data, pdf_path)
                charts = self._extract_charts_from_json(json_data, pdf_path)
            
            # Step 2: Extract text with Unstructured for DOC index
            unstructured_text = self._extract_text_with_unstructured(pdf_path)
            text_elements.extend(unstructured_text)
            
            print(f"    ✓ Extracted: {len(text_elements)} text elements, {len(tables)} tables, {len(charts)} charts")
            
        except Exception as e:
            print(f"    ⚠ LlamaParse error: {e}")
            import traceback
            traceback.print_exc()
        
        return ParsedDocument(
            pdf_path=str(pdf_path),
            text_elements=text_elements,
            tables=tables,
            charts=charts,
        )

    def parse_directory(self, pdf_dir: Path) -> List[ParsedDocument]:
        """Parse all PDFs in a directory."""
        docs: List[ParsedDocument] = []
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        total = len(pdf_files)
        print(f"  Found {total} PDF files in {pdf_dir.name}/")
        for idx, pdf_path in enumerate(pdf_files, 1):
            print(f"\n  [{idx}/{total}] Parsing: {pdf_path.name}")
            doc = self.parse_pdf(pdf_path)
            tables_count = len(doc.tables)
            text_count = len(doc.text_elements)
            charts_count = len(doc.charts)
            print(f"  ✓ Completed: {text_count} text elements, {tables_count} tables, {charts_count} charts")
            docs.append(doc)
        return docs


def parse_all_artifacts(base_dir: Path) -> List[ParsedDocument]:
    """
    Convenience function: parse all PDFs under artifacts/1 and artifacts/2.
    """
    parser = PDFParser()
    all_docs: List[ParsedDocument] = []

    for sub in ("1", "2"):
        pdf_dir = base_dir / "artifacts" / sub
        if pdf_dir.is_dir():
            all_docs.extend(parser.parse_directory(pdf_dir))

    return all_docs
