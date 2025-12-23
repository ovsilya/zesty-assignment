from __future__ import annotations

"""
Table processing:
- Convert extracted tables (from LlamaParse JSON) to DataFrames
- Normalize/clean basic structure
- Generate short LLM summaries for semantic indexing
"""

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings

import pandas as pd

from .pdf_parser import ParsedDocument, ParsedTable

# Suppress deprecation warnings for VertexAI LLM wrapper from langchain-google-vertexai.
# These classes still work today; we'll migrate to langchain-google-genai in a future refactor.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="langchain_google_vertexai",
)

try:
    from langchain_google_vertexai import VertexAI
    from langchain_core.prompts import PromptTemplate
except Exception:  # pragma: no cover - optional dependency
    VertexAI = None  # type: ignore
    PromptTemplate = None  # type: ignore


@dataclass
class ProcessedTable:
    pdf_path: str
    table_id: str
    page_number: Optional[int]
    dataframe: pd.DataFrame
    summary: str


class TableProcessor:
    """Turns ParsedTable instances into cleaned DataFrames plus LLM summaries."""

    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        if VertexAI is None or PromptTemplate is None:
            raise ImportError(
                "langchain-google-vertexai is required for TableProcessor. "
                "Install it and configure Vertex AI credentials."
            )
        self.llm = VertexAI(model_name=model_name)
        self.summary_prompt = PromptTemplate(
            input_variables=["table", "context"],
            template=(
                "You are summarizing a table extracted from an insurance filing PDF.\n"
                "Provide a concise 2–3 sentence summary that includes:\n"
                "1) What the table describes (subject and granularity),\n"
                "2) Key columns/dimensions, and\n"
                "3) Any notable categories or ranges.\n\n"
                "Context: {context}\n\n"
                "Table (markdown):\n{table}\n\n"
                "Summary:"
            ),
        )

    def _to_dataframe(self, table: ParsedTable) -> pd.DataFrame:
        """Convert ParsedTable to DataFrame using rows from LlamaParse JSON.
        
        Uses "rows" from items with type="table".
        Header handling:
        - If isPerfectTable is explicitly False and there are exactly 2 columns,
          treat as a key-value table with generic headers col1, col2.
        - Otherwise, use a simple heuristic to decide if the first row is a header:
          * first row should be mostly text (has letters)
          * second row should be mostly numeric/currency/percent
          If the heuristic passes, use first row as headers, else use col1, col2, ...
        """
        if not table.rows or len(table.rows) == 0:
            return pd.DataFrame()
        
        try:
            rows = table.rows
            if not all(isinstance(row, list) for row in rows):
                raise ValueError("Rows must be a list of lists")
            
            # Normalize to strings
            str_rows = [["" if v is None else str(v) for v in row] for row in rows]
            first = str_rows[0]
            n_cols = len(first)
            
            # Helper detectors
            def has_alpha(s: str) -> bool:
                return any(ch.isalpha() for ch in s)
            
            def is_numericish(s: str) -> bool:
                s2 = s.strip().replace(",", "").replace("$", "").replace("%", "")
                if not s2:
                    return False
                return all(ch.isdigit() or ch in ".-" for ch in s2)
            
            is_perfect = table.metadata.get("isPerfectTable")
            
            # 1) Key-value style: explicitly imperfect, exactly 2 columns -> generic headers
            if is_perfect is False and n_cols == 2:
                df = pd.DataFrame(str_rows)
                df.columns = [f"col{i+1}" for i in range(len(df.columns))]
                return df
            
            # 2) Heuristic header detection for all other tables
            if len(str_rows) >= 2:
                second = str_rows[1]
                header_alpha = sum(1 for v in first if has_alpha(v))
                data_numeric = sum(1 for v in second if is_numericish(v))
                header_alpha_ratio = header_alpha / max(n_cols, 1)
                data_numeric_ratio = data_numeric / max(len(second), 1)
                
                use_header = header_alpha_ratio >= 0.5 and data_numeric_ratio >= 0.5
                
                if use_header:
                    headers = [v if v else f"col{i+1}" for i, v in enumerate(first)]
                    df = pd.DataFrame(str_rows[1:], columns=headers)
                else:
                    df = pd.DataFrame(str_rows)
                    df.columns = [f"col{i+1}" for i in range(len(df.columns))]
            else:
                # Only one row: cannot reliably detect headers -> generic
                df = pd.DataFrame(str_rows)
                df.columns = [f"col{i+1}" for i in range(len(df.columns))]
            
            return df
        except Exception as e:
            print(f"        ⚠ Rows parsing failed: {e}")
            return pd.DataFrame()

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic cleanup: drop empty rows/cols, strip headers."""
        if df.empty:
            return df

        df = df.copy()
        df = df.dropna(how="all").dropna(axis=1, how="all")
        df.columns = [str(c).strip() for c in df.columns]
        return df

    def _summarize(self, df: pd.DataFrame, context: str) -> str:
        if df.empty:
            return "Empty or unparseable table."

        table_md = df.to_markdown(index=False)
        return self.llm.invoke(
            self.summary_prompt.format(table=table_md, context=context)
        )

    def process_documents(self, docs: List[ParsedDocument]) -> List[ProcessedTable]:
        processed: List[ProcessedTable] = []
        
        total_tables = sum(len(doc.tables) for doc in docs)
        current = 0
        
        for doc in docs:
            pdf_name = Path(doc.pdf_path).name
            for idx, table in enumerate(doc.tables):
                current += 1
                parser_name = table.metadata.get("parser", "unknown")
                print(f"  [{current}/{total_tables}] Processing table from {pdf_name} (page {table.page_number or '?'}, parser: {parser_name})...")
                
                # Step 1: Convert to DataFrame (using structured data from LlamaParse JSON)
                data_format = "rows" if table.rows else "CSV"
                print(f"    → Converting {data_format} to Pandas DataFrame...", end=" ", flush=True)
                df = self._to_dataframe(table)
                if df.empty:
                    print(f"⚠ (empty {data_format}, skipping)")
                    continue
                print(f"✓ ({len(df)} rows × {len(df.columns)} columns)")
                
                # Step 2: Normalize DataFrame
                print(f"    → Normalizing DataFrame (removing empty rows/cols, cleaning headers)...", end=" ", flush=True)
                df_before = len(df)
                df = self._normalize(df)
                df_after = len(df)
                if df.empty:
                    print("⚠ (empty after normalization, skipping)")
                    continue
                removed = df_before - df_after
                if removed > 0:
                    print(f"✓ (removed {removed} empty rows, {len(df.columns)} columns remain)")
                else:
                    print(f"✓ ({len(df.columns)} columns)")
                
                # Step 3: Convert to Markdown for LLM
                print(f"    → Converting DataFrame to Markdown format for LLM...", end=" ", flush=True)
                table_md = df.to_markdown(index=False)
                md_size = len(table_md)
                print(f"✓ ({md_size:,} characters)")
                
                # Step 4: Generate LLM summary
                context = f"From document {pdf_name}, page {table.page_number or '?'}."
                print(f"    → Generating LLM summary with Gemini 2.5 Flash...", end=" ", flush=True)
                summary = self._summarize(df, context=context)
                summary_preview = summary[:80] + "..." if len(summary) > 80 else summary
                print(f"✓")
                print(f"      Summary preview: {summary_preview}")

                processed.append(
                    ProcessedTable(
                        pdf_path=doc.pdf_path,
                        table_id=f"{Path(doc.pdf_path).stem}_table_{idx}",
                        page_number=table.page_number,
                        dataframe=df,
                        summary=summary,
                    )
                )

        return processed


