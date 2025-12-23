"""
Data Processing and Indexing Pipeline

This script processes each PDF completely before moving to the next:
1. Parse PDF
2. Save extracted tables as CSV files (for inspection/comparison)
3. Process tables (normalization + LLM summaries)
4. Add to DOC Index (vector store) in BigQuery
5. Store FACTS tables in BigQuery

Then moves to next PDF.

Features:
- Tracks processed PDFs in a log file to avoid re-processing (saves LlamaParse API calls)
- Saves extracted tables as CSV files for inspection and comparison

Run this ONCE to build all indices. Then use query_rag.py to test queries.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd

from src.parsing.pdf_parser import PDFParser
from src.parsing.table_processor import TableProcessor
from src.indexing.vector_store import VectorIndexBuilder
from src.indexing.facts_store import FactsStoreBuilder


def load_processing_log(log_file: Path) -> dict:
    """Load the processing log file."""
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_processing_log(log_file: Path, log_data: dict) -> None:
    """Save the processing log file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)


def is_pdf_processed(pdf_path: Path, log_data: dict) -> bool:
    """Check if a PDF has already been processed.
    
    If the environment variable FORCE_REPROCESS is set to a truthy value
    (\"1\", \"true\", \"yes\"), this always returns False so that PDFs are
    reprocessed and CSV/BigQuery tables are regenerated.
    """
    # Optional override to force reprocessing
    force_reprocess = os.getenv("FORCE_REPROCESS", "").lower() in ("1", "true", "yes")
    if force_reprocess:
        return False
    pdf_key = str(pdf_path.resolve())
    if pdf_key in log_data:
        entry = log_data[pdf_key]
        # Check if processing was successful
        if entry.get('status') == 'completed':
            return True
    return False


def save_extracted_tables_csv(parsed_doc, output_dir: Path) -> int:
    """Save extracted tables as CSV files (before processing/normalization).
    
    Uses only "rows" from LlamaParse JSON and mirrors header handling in TableProcessor:
    - If isPerfectTable is explicitly False and there are exactly 2 columns,
      treat as a key-value table with generic headers col1, col2.
    - Otherwise, use a simple heuristic to decide if the first row is a header:
      * first row should be mostly text (has letters)
      * second row should be mostly numeric/currency/percent
      If the heuristic passes, use first row as headers, else use col1, col2, ...
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_stem = Path(parsed_doc.pdf_path).stem
    saved_count = 0
    
    for idx, table in enumerate(parsed_doc.tables):
        try:
            page_num = table.page_number or '?'
            csv_path = output_dir / f"{pdf_stem}_table_{idx}_page_{page_num}.csv"
            if not table.rows or len(table.rows) == 0:
                continue

            rows = table.rows
            # Normalize to strings
            str_rows = [["" if v is None else str(v) for v in row] for row in rows]
            first = str_rows[0]
            n_cols = len(first)

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
            else:
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

            if df is not None and not df.empty:
                df.to_csv(csv_path, index=False)
                saved_count += 1
        except Exception as e:
            print(f"        ⚠ Failed to convert/save table {idx}: {e}")
    
    return saved_count


def update_processing_log(
    log_data: dict,
    pdf_path: Path,
    status: str,
    tables_count: int,
    tables_stored: int,
    text_count: int = 0,
    charts_count: int = 0,
    error: str = None
) -> None:
    """Update the processing log with PDF processing information."""
    pdf_key = str(pdf_path.resolve())
    log_data[pdf_key] = {
        'pdf_name': pdf_path.name,
        'status': status,
        'processed_date': datetime.now().isoformat(),
        'text_elements': text_count,
        'tables_extracted': tables_count,
        'tables_stored': tables_stored,
        'charts_extracted': charts_count,
    }
    if error:
        log_data[pdf_key]['error'] = str(error)


def main() -> None:
    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise RuntimeError("Missing GOOGLE_CLOUD_PROJECT environment variable.")

    base_dir = Path(__file__).resolve().parent
    dataset_name = "rag_dataset"
    
    # Setup directories
    log_file = base_dir / "processing_log.json"
    tables_output_dir = base_dir / "extracted_tables"
    
    # Load processing log
    log_data = load_processing_log(log_file)
    print(f"\n📋 Processing log: {log_file}")
    print(f"   Found {len(log_data)} previously processed PDFs")

    # Initialize processors and builders
    parser = PDFParser()
    table_processor = TableProcessor()
    vec_builder = VectorIndexBuilder(project_id=project_id, dataset_name=dataset_name)
    facts_builder = FactsStoreBuilder(project_id=project_id, dataset_name=dataset_name)
    
    # Get or create vector store (for incremental additions)
    print("\n" + "="*60)
    print("INITIALIZING VECTOR STORE")
    print("="*60)
    vector_store = vec_builder.get_or_create_store()
    print("✓ Vector store initialized/connected")

    # Collect all PDFs
    pdf_files = []
    for sub in ("1", "2"):
        pdf_dir = base_dir / "artifacts" / sub
        if pdf_dir.is_dir():
            pdf_files.extend(sorted(pdf_dir.glob("*.pdf")))
    
    total_pdfs = len(pdf_files)
    print(f"\nFound {total_pdfs} PDF files to process")
    
    # Process each PDF completely before moving to the next
    total_tables_processed = 0
    total_tables_stored = 0
    skipped_count = 0
    
    for idx, pdf_path in enumerate(pdf_files, 1):
        print("\n" + "="*60)
        print(f"PDF {idx}/{total_pdfs}: {pdf_path.name}")
        print("="*60)
        
        # Check if already processed
        if is_pdf_processed(pdf_path, log_data):
            print(f"⏭️  Skipping (already processed)")
            entry = log_data[str(pdf_path.resolve())]
            print(f"   Previously processed: {entry.get('processed_date', 'unknown')}")
            print(f"   Tables: {entry.get('tables_extracted', 0)} extracted, {entry.get('tables_stored', 0)} stored")
            skipped_count += 1
            continue
        
        try:
            # Step 1: Parse PDF (LlamaParse only - text, tables, charts)
            print(f"\n[1/5] Parsing PDF with LlamaParse...")
            parsed_doc = parser.parse_pdf(pdf_path)
            print(f"✓ Parsed: {len(parsed_doc.text_elements)} text elements, {len(parsed_doc.tables)} tables, {len(parsed_doc.charts)} charts")
            
            # Step 2: Save extracted tables as CSV (before processing)
            print(f"\n[2/5] Saving extracted tables as CSV files...")
            csv_count = save_extracted_tables_csv(parsed_doc, tables_output_dir)
            print(f"✓ Saved {csv_count} tables as CSV files to: {tables_output_dir}")
            
            # Step 3: Process tables
            print(f"\n[3/5] Processing tables (normalization + LLM summaries)...")
            processed_tables = table_processor.process_documents([parsed_doc])
            print(f"✓ Processed {len(processed_tables)} tables")
            total_tables_processed += len(processed_tables)
            
            # Step 4: Add to vector index (text, tables, charts)
            print(f"\n[4/5] Adding to DOC Index (vector store)...")
            vec_builder.add_document_to_index(parsed_doc, processed_tables, vector_store)
            print(f"✓ Added to vector index")
            
            # Step 5: Store FACTS tables in BigQuery
            print(f"\n[5/5] Storing FACTS tables in BigQuery...")
            stored_info = facts_builder.store_all(processed_tables)
            stored_count = len(stored_info)
            total_tables_stored += stored_count
            print(f"✓ Stored {stored_count} tables in BigQuery")
            
            # Update processing log
            update_processing_log(
                log_data, pdf_path, 'completed',
                len(parsed_doc.tables), stored_count,
                len(parsed_doc.text_elements), len(parsed_doc.charts)
            )
            save_processing_log(log_file, log_data)
            
            print(f"\n✓ Completed PDF {idx}/{total_pdfs}: {pdf_path.name}")
            print(f"  - Text elements: {len(parsed_doc.text_elements)}")
            print(f"  - Tables extracted: {len(parsed_doc.tables)}")
            print(f"  - Tables processed: {len(processed_tables)}")
            print(f"  - Tables stored: {stored_count}")
            print(f"  - Charts extracted: {len(parsed_doc.charts)}")
            
        except Exception as e:
            print(f"\n✗ Error processing {pdf_path.name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Update log with error
            update_processing_log(
                log_data, pdf_path, 'error',
                len(parsed_doc.tables) if 'parsed_doc' in locals() else 0,
                0,
                len(parsed_doc.text_elements) if 'parsed_doc' in locals() else 0,
                len(parsed_doc.charts) if 'parsed_doc' in locals() else 0,
                error=str(e)
            )
            save_processing_log(log_file, log_data)

    print("\n" + "="*60)
    print("✓ DATA PROCESSING COMPLETE!")
    print("="*60)
    print(f"  - Total PDFs: {total_pdfs}")
    print(f"  - Processed: {total_pdfs - skipped_count}")
    print(f"  - Skipped (already processed): {skipped_count}")
    print(f"  - Total tables processed: {total_tables_processed}")
    print(f"  - Total tables stored: {total_tables_stored}")
    print(f"  - Vector store: {project_id}.{dataset_name}.doc_index")
    print(f"  - FACTS tables: {project_id}.{dataset_name}.table_*")
    print(f"  - Extracted tables CSV: {tables_output_dir}")
    print(f"  - Processing log: {log_file}")
    print("\nNext steps:")
    print("  - Run queries: python3 query_rag.py")
    print("  - Or run evaluation: python3 -m src.evaluation.evaluate")


if __name__ == "__main__":
    main()

