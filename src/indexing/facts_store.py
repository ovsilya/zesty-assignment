from __future__ import annotations

"""
FACTS Store: structured tables in BigQuery for precise SQL queries.
"""

from typing import List, Dict

import pandas as pd

from src.parsing.table_processor import ProcessedTable


try:
    from google.cloud import bigquery
except Exception:  # pragma: no cover
    bigquery = None  # type: ignore


class FactsStoreBuilder:
    """Uploads processed tables into BigQuery tables."""

    def __init__(self, project_id: str, dataset_name: str = "rag_dataset") -> None:
        if bigquery is None:
            raise ImportError(
                "google-cloud-bigquery is required for FactsStoreBuilder."
            )

        self.client = bigquery.Client(project=project_id)
        self.dataset_name = dataset_name
        self._ensure_dataset()

    def _ensure_dataset(self) -> None:
        dataset_id = f"{self.client.project}.{self.dataset_name}"
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"
        self.client.create_dataset(dataset, exists_ok=True)

    def _sanitize_column_name(self, col_name: str) -> str:
        """
        Sanitize column name for BigQuery compliance.
        - Must contain only letters, numbers, underscores
        - Must start with letter or underscore
        - Max 300 characters
        - Cannot start with reserved prefixes: _PARTITION, _TABLE_, _FILE_, _ROW_TIMESTAMP, __ROOT__, _COLIDENTIFIER
        """
        # Handle multi-level headers (tuples)
        if isinstance(col_name, tuple):
            # Flatten tuple to string, join with underscore
            col_name = "_".join(str(c) for c in col_name if c)
        
        col_name = str(col_name)
        
        # Remove or replace invalid characters
        sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in col_name)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")
        
        # Ensure it starts with letter or underscore (not number)
        if sanitized and sanitized[0].isdigit():
            sanitized = "col_" + sanitized
        
        # Ensure it doesn't start with reserved prefixes
        reserved_prefixes = ["_PARTITION", "_TABLE_", "_FILE_", "_ROW_TIMESTAMP", "__ROOT__", "_COLIDENTIFIER"]
        upper_sanitized = sanitized.upper()
        for prefix in reserved_prefixes:
            if upper_sanitized.startswith(prefix):
                sanitized = "meta_" + sanitized
                break
        
        # Limit length to 300 characters
        sanitized = sanitized[:300]
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = "unnamed_col"
        
        return sanitized

    def _flatten_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten MultiIndex columns and ensure all values are scalar."""
        df = df.copy()
        
        # Flatten MultiIndex columns first (before any cell access)
        if isinstance(df.columns, pd.MultiIndex):
            # Join multi-level column names with underscore
            new_columns = []
            for col in df.columns:
                if isinstance(col, tuple):
                    # Join tuple elements with underscore
                    col_str = '_'.join(str(c) for c in col if c).strip('_')
                else:
                    col_str = str(col)
                new_columns.append(col_str)
            df.columns = pd.Index(new_columns)
        
        # Ensure all column names are strings (not tuples or other objects)
        df.columns = [str(col) for col in df.columns]
        
        # Remove any duplicate column names by appending numbers
        # This prevents issues where df[col] returns a DataFrame instead of Series
        seen = {}
        new_cols = []
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                new_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_cols.append(col)
        df.columns = new_cols
        
        # Check for non-scalar values more efficiently
        # Only process columns that might have issues (object dtype)
        for col in df.columns:
            try:
                col_series = df[col]
                # Ensure we have a Series, not a DataFrame
                if isinstance(col_series, pd.DataFrame):
                    # If it's a DataFrame, flatten it further
                    df = df.drop(columns=[col])
                    for subcol in col_series.columns:
                        df[f"{col}_{subcol}"] = col_series[subcol]
                    continue
                
                if col_series.dtype == 'object':
                    # Check a sample to see if conversion is needed
                    sample = col_series.dropna().head(10)
                    if len(sample) > 0:
                        # Check if any values are non-scalar
                        has_nested = any(
                            isinstance(val, (pd.DataFrame, pd.Series, list, dict))
                            for val in sample
                        )
                        if has_nested:
                            # Convert entire column to string
                            df[col] = col_series.apply(
                                lambda x: str(x) if pd.notna(x) and isinstance(x, (pd.DataFrame, pd.Series, list, dict)) else x
                            )
            except Exception as e:
                # If we can't process this column, convert it to string
                try:
                    df[col] = df[col].astype(str)
                except:
                    # Last resort: drop the problematic column
                    df = df.drop(columns=[col])
        
        return df

    def _sanitize_dataframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sanitize all column names in a DataFrame for BigQuery."""
        df = self._flatten_dataframe(df)
        
        # Create mapping of old to new column names
        column_mapping = {}
        used_names = set()
        
        for col in df.columns:
            new_col = self._sanitize_column_name(col)
            # Handle duplicates by appending number
            original_new = new_col
            counter = 1
            while new_col in used_names:
                new_col = f"{original_new}_{counter}"
                counter += 1
            used_names.add(new_col)
            column_mapping[col] = new_col
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        return df

    def store_table(self, table: ProcessedTable) -> str:
        """Store a single table as a BigQuery table."""
        # Sanitize table ID: BigQuery allows only letters, numbers, underscores
        # Must start with letter or underscore, max 1024 chars
        sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in table.table_id)
        # Ensure it starts with letter or underscore
        if sanitized and not (sanitized[0].isalpha() or sanitized[0] == "_"):
            sanitized = "t_" + sanitized
        # Limit length and ensure uniqueness
        table_name = f"table_{sanitized[:200]}"  # BigQuery limit is 1024, but keep it reasonable
        full_id = f"{self.client.project}.{self.dataset_name}.{table_name}"

        df = table.dataframe.copy()
        
        # Validate DataFrame is not empty
        if df.empty:
            raise ValueError(f"DataFrame for table {table.table_id} is empty")
        
        # Add metadata columns with safe names (avoid reserved prefixes)
        df["meta_source_pdf"] = str(table.pdf_path)
        df["meta_table_id"] = str(table.table_id)
        df["meta_page_number"] = int(table.page_number) if table.page_number is not None else None
        
        # Sanitize all column names for BigQuery compliance
        df = self._sanitize_dataframe_columns(df)
        
        # Final validation: ensure DataFrame is still valid
        if df.empty:
            raise ValueError(f"DataFrame for table {table.table_id} became empty after sanitization")
        
        # Reset index to avoid index column issues
        df = df.reset_index(drop=True)
        
        # Ensure all dtypes are BigQuery-compatible
        # Convert object columns to string if they contain non-scalar values
        for col in df.columns:
            try:
                col_series = df[col]
                # Ensure we have a Series, not a DataFrame
                if isinstance(col_series, pd.DataFrame):
                    # This shouldn't happen after flattening, but handle it
                    # Flatten the DataFrame column
                    for subcol_idx, subcol in enumerate(col_series.columns):
                        df[f"{col}_sub_{subcol_idx}"] = col_series[subcol]
                    df = df.drop(columns=[col])
                    continue
                
                if col_series.dtype == 'object':
                    # Check if column contains non-scalar values
                    try:
                        # Try to convert to string, which BigQuery can handle
                        df[col] = col_series.astype(str)
                    except Exception:
                        # If conversion fails, replace problematic values
                        df[col] = col_series.apply(lambda x: str(x) if pd.notna(x) else None)
            except Exception as e:
                # If we can't process this column, try to convert or drop it
                try:
                    df[col] = df[col].astype(str)
                except:
                    # Last resort: drop the problematic column
                    print(f"        ⚠ Warning: Dropping problematic column '{col}' due to: {e}")
                    df = df.drop(columns=[col])

        try:
            # Delete existing table if it exists (to overwrite)
            try:
                self.client.delete_table(full_id, not_found_ok=True)
            except Exception:
                pass  # Ignore deletion errors
            
            # Load DataFrame to BigQuery
            job = self.client.load_table_from_dataframe(df, full_id)
            job.result()  # Wait for job to complete
            return full_id
        except Exception as e:
            # Provide more detailed error information
            error_msg = f"Failed to upload table {table.table_id} to BigQuery: {e}"
            error_msg += f"\n  DataFrame shape: {df.shape}"
            error_msg += f"\n  Columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}"
            try:
                # Safely get column types
                dtypes_dict = dict(df.dtypes)
                error_msg += f"\n  Column types: {dtypes_dict}"
            except Exception:
                error_msg += f"\n  Column types: (could not retrieve)"
            raise RuntimeError(error_msg) from e

    def store_all(self, tables: List[ProcessedTable]) -> List[Dict[str, str]]:
        stored: List[Dict[str, str]] = []
        total = len(tables)
        print(f"  - Uploading {total} tables to BigQuery...")
        for idx, t in enumerate(tables, 1):
            try:
                print(f"  [{idx}/{total}] Storing {t.table_id}...", end=" ", flush=True)
                full_id = self.store_table(t)
                print("✓")
                stored.append(
                    {
                        "table_id": t.table_id,
                        "bigquery_table": full_id,
                        "pdf_path": t.pdf_path,
                    }
                )
            except Exception as exc:  # pragma: no cover - logging path
                print(f"✗ Error: {exc}")
        return stored


