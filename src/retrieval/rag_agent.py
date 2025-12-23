from __future__ import annotations

"""
Hybrid RAG agent:
- Classifies queries (semantic / quantitative / hybrid)
- Vector retrieval from DOC Index + SQL over FACTS Store
- Reranking and final answer generation with source attribution
"""

from typing import List, Dict, Any
from pathlib import Path

from langchain_google_vertexai import VertexAI
import warnings
# Suppress deprecation warning for VertexAI (still works, will be removed in v4.0)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_google_vertexai")

from .utils import (
    extract_table_ids_from_results,
    extract_specific_values_from_query,
    extract_geographic_terms,
)
from .retrieval_logic import RetrievalLogic


try:
    from langchain_google_community import BigQueryVectorStore
    from langchain_google_vertexai import VertexAIEmbeddings
    from google.cloud import bigquery
    from flashrank import Ranker, RerankRequest
except Exception:  # pragma: no cover
    BigQueryVectorStore = None  # type: ignore
    VertexAIEmbeddings = None  # type: ignore
    bigquery = None  # type: ignore
    Ranker = None  # type: ignore
    RerankRequest = None  # type: ignore


class RAGAgent:
    """Implements the hybrid retrieval and answer synthesis."""

    def __init__(
        self,
        project_id: str,
        dataset_name: str = "rag_dataset",
        table_name: str = "doc_index",
        llm_model: str = "gemini-2.5-pro",
        location: str = "US",
    ) -> None:
        if (
            BigQueryVectorStore is None
            or VertexAIEmbeddings is None
            or bigquery is None
            or Ranker is None
            or RerankRequest is None
        ):
            raise ImportError(
                "RAGAgent requires langchain-google-vertexai, "
                "langchain-google-community, google-cloud-bigquery and flashrank."
            )

        # Initialize LLM with max_output_tokens to prevent truncation
        # Gemini 2.5 Pro supports up to 8192 output tokens (default is 2048)
        # Note: reasoning_budget is not directly supported in VertexAI wrapper
        # The model will use its default reasoning capabilities
        self.llm = VertexAI(
            model_name=llm_model,
            max_output_tokens=8192,  # Maximum for Gemini 2.5 Pro to prevent truncation
            temperature=0.1,  # Lower temperature for more consistent, factual responses
        )
        self.project_id = project_id
        self.dataset_name = dataset_name
        self.table_name = table_name

        self.embeddings = VertexAIEmbeddings(model_name="text-embedding-005")
        self.vector_store = BigQueryVectorStore(
            project_id=project_id,
            dataset_name=dataset_name,
            table_name=table_name,
            location=location,
            embedding=self.embeddings,
        )

        self.bq_client = bigquery.Client(project=project_id)
        
        # Load prompt template (using v2 - generic version without hardcoded examples)
        prompt_template_path = Path(__file__).parent / "prompt_template_v2.txt"
        if not prompt_template_path.exists():
            # Fallback to v1 if v2 doesn't exist
            prompt_template_path = Path(__file__).parent / "prompt_template.txt"
            if not prompt_template_path.exists():
                raise FileNotFoundError(
                    f"Prompt template file not found: {prompt_template_path}\n"
                    "Please ensure prompt_template_v2.txt or prompt_template.txt exists in src/retrieval/"
                )
        self.prompt_template = prompt_template_path.read_text(encoding='utf-8')
        
        # Initialize FlashRank ranker (optional - will skip reranking if it fails)
        try:
            self.ranker = Ranker()
            self.ranker_available = True
        except Exception as e:
            print(f"⚠ Warning: FlashRank ranker initialization failed: {e}")
            print("   Continuing without reranking (using original retrieval order)")
            self.ranker = None
            self.ranker_available = False
        
        # Initialize retrieval logic module
        self.retrieval_logic = RetrievalLogic(
            bq_client=self.bq_client,
            project_id=project_id,
            dataset_name=dataset_name,
            table_name=table_name,
            vector_store=self.vector_store,
            ranker=self.ranker,
            ranker_available=self.ranker_available,
        )

    # ----- Routing and retrieval -------------------------------------------------

    def _extract_table_ids_from_results(self, semantic_results: List[Dict[str, Any]]) -> List[str]:
        """Extract table_ids from semantic retrieval results that are table summaries."""
        return extract_table_ids_from_results(semantic_results)
    
    def diagnose_table_content(self, query: str) -> Dict[str, Any]:
        """Diagnostic function to inspect what tables contain relevant data for a query.
        
        This helps debug why certain data isn't being found.
        
        Args:
            query: The user's question
            
        Returns:
            Dictionary with diagnostic information about relevant tables
        """
        query_lower = query.lower()
        
        # Extract search terms
        search_terms = []
        if "base rate" in query_lower:
            search_terms.extend(["base", "rate", "ho3"])
        if "territory" in query_lower or "zip" in query_lower:
            search_terms.extend(["territory", "zip"])
        if "comprehensive" in query_lower:
            search_terms.extend(["comprehensive"])
        if "grg" in query_lower or "collision" in query_lower:
            search_terms.extend(["grg", "collision", "motorcycle", "rating", "group"])
        if "hurricane" in query_lower:
            search_terms.extend(["hurricane", "deductible", "factor", "coast"])
        
        # Search by column names
        column_keywords = []
        if "rate" in query_lower:
            column_keywords.extend(["rate", "base", "premium"])
        if "territory" in query_lower:
            column_keywords.extend(["territory", "zip", "code"])
        if "comprehensive" in query_lower:
            column_keywords.extend(["comprehensive"])
        if "grg" in query_lower:
            column_keywords.extend(["grg", "rating", "group"])
        if "hurricane" in query_lower:
            column_keywords.extend(["hurricane", "deductible"])
        
        column_matches = self._search_tables_by_column_name(column_keywords) if column_keywords else []
        
        # Inspect top matching tables
        inspected_tables = []
        for table_id in column_matches[:10]:
            inspection = self._inspect_table_content(table_id, max_rows=5)
            inspected_tables.append(inspection)
        
        return {
            "query": query,
            "search_terms": search_terms,
            "column_keywords": column_keywords,
            "tables_with_matching_columns": column_matches[:20],
            "inspected_tables": inspected_tables,
        }

    def _semantic_retrieval(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic retrieval using the retrieval logic module."""
        return self.retrieval_logic.semantic_retrieval(query, k=k)
    
    def _expand_by_page_and_document(self, initial_results: List[Dict[str, Any]], max_expansion: int = 200) -> List[Dict[str, Any]]:
        """Expand retrieval to include all content from same pages and documents."""
        return self.retrieval_logic.expand_by_page_and_document(initial_results, max_expansion=max_expansion)
        """Expand retrieval to include all content from same pages and documents.
        
        When a text chunk or table is found, retrieve ALL:
        - Text chunks from the same page
        - Tables from the same page
        - All content from the same document (if needed)
        
        Uses direct BigQuery queries to find all content from same source+page.
        
        Args:
            initial_results: Initial semantic retrieval results
            max_expansion: Maximum number of additional items to retrieve
            
        Returns:
            Expanded list of results with all content from same pages/documents
        """
        if not initial_results:
            return []
        
        # Collect unique (source, page_number) pairs from initial results
        source_page_pairs = set()
        sources = set()
        
        for result in initial_results:
            meta = result.get("metadata", {})
            source = meta.get("source")
            page_num = meta.get("page_number")
            
            if source:
                sources.add(source)
                if page_num is not None:
                    source_page_pairs.add((source, page_num))
        
        # Retrieve all content from these pages using direct BigQuery query
        expanded_results = list(initial_results)  # Start with initial results
        seen_content = {r["content"] for r in initial_results}  # Track what we've already added
        
        # Query BigQuery directly for all content from same pages
        try:
            for source, page_num in list(source_page_pairs)[:15]:  # Limit to avoid too many queries
                if len(expanded_results) >= max_expansion:
                    break
                
                # Query BigQuery vector store table for all documents from same source and page
                sql = f"""
                SELECT 
                    content,
                    source,
                    page_number,
                    element_type,
                    table_id,
                    chunk_id
                FROM `{self.project_id}.{self.dataset_name}.{self.table_name}`
                WHERE source = @source
                  AND page_number = @page_num
                LIMIT 100
                """
                
                job_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("source", "STRING", source),
                        bigquery.ScalarQueryParameter("page_num", "INT64", int(page_num)),
                    ]
                )
                
                try:
                    job = self.bq_client.query(sql, job_config=job_config)
                    rows = list(job.result())
                    
                    for row in rows:
                        # Reconstruct document-like structure
                        doc_content = row.get("content", "")
                        if not doc_content or doc_content in seen_content:
                            continue
                        
                        doc_meta = {
                            "source": row.get("source"),
                            "page_number": row.get("page_number"),
                            "element_type": row.get("element_type", "text"),
                        }
                        
                        if row.get("table_id"):
                            doc_meta["table_id"] = row.get("table_id")
                        if row.get("chunk_id") is not None:
                            doc_meta["chunk_id"] = row.get("chunk_id")
                        
                        expanded_results.append({
                            "content": doc_content,
                            "score": 0.8,  # Lower score for expanded items
                            "metadata": doc_meta,
                            "expanded": True,  # Mark as expanded
                        })
                        seen_content.add(doc_content)
                        
                        if len(expanded_results) >= max_expansion:
                            break
                except Exception as e:
                    # Fallback: try without parameters (string matching)
                    try:
                        sql_simple = f"""
                        SELECT 
                            content,
                            source,
                            page_number,
                            element_type,
                            table_id,
                            chunk_id
                        FROM `{self.project_id}.{self.dataset_name}.{self.table_name}`
                        WHERE source LIKE '%{source.split("/")[-1]}%'
                          AND CAST(page_number AS INT64) = {int(page_num)}
                        LIMIT 100
                        """
                        job = self.bq_client.query(sql_simple)
                        rows = list(job.result())
                        
                        for row in rows:
                            doc_content = row.get("content", "")
                            if not doc_content or doc_content in seen_content:
                                continue
                            
                            doc_meta = {
                                "source": row.get("source"),
                                "page_number": row.get("page_number"),
                                "element_type": row.get("element_type", "text"),
                            }
                            
                            if row.get("table_id"):
                                doc_meta["table_id"] = row.get("table_id")
                            
                            expanded_results.append({
                                "content": doc_content,
                                "score": 0.8,
                                "metadata": doc_meta,
                                "expanded": True,
                            })
                            seen_content.add(doc_content)
                            
                            if len(expanded_results) >= max_expansion:
                                break
                    except Exception:
                        continue
        
        except Exception as e:
            print(f"      ⚠ Error expanding by page/document: {e}")
        
        # Sort by score (original results first, then expanded)
        expanded_results.sort(key=lambda x: x["score"], reverse=True)
        return expanded_results

    def _list_tables(self) -> List[str]:
        dataset_ref = self.bq_client.dataset(self.dataset_name)
        tables = list(self.bq_client.list_tables(dataset_ref))
        # Filter out the vector index table
        return [t.table_id for t in tables if t.table_id != self.table_name]
    
    def _search_tables_by_column_name(self, column_keywords: List[str]) -> List[str]:
        """Search for tables that contain columns matching the given keywords."""
        return self.retrieval_logic.search_tables_by_column_name(column_keywords)
    
    def _search_tables_by_content(self, search_terms: List[str], sample_rows: int = 20, max_tables_to_search: int = 200, query_context: str = "") -> List[str]:
        """Search for tables that contain specific values in their data."""
        return self.retrieval_logic.search_tables_by_content(
            search_terms, 
            sample_rows=sample_rows, 
            max_tables_to_search=max_tables_to_search, 
            query_context=query_context
        )
    
    def _inspect_table_content(self, table_id: str, max_rows: int = 10) -> Dict[str, Any]:
        """Inspect a table's content and return summary information."""
        return self.retrieval_logic.inspect_table_content(table_id, max_rows=max_rows)

    def _get_table_schema(self, table_id: str) -> Dict[str, Any]:
        """Get schema (column names and types) for a BigQuery table."""
        return self.retrieval_logic.get_table_schema(table_id)
    
    def _find_bigquery_table_by_table_id(self, table_id: str) -> str | None:
        """Find the BigQuery table name that corresponds to a given table_id."""
        return self.retrieval_logic.find_bigquery_table_by_table_id(table_id)
    
    def _get_relevant_table_schemas(self, query: str, table_ids_from_semantic: List[str] = None, max_tables: int = 15) -> List[Dict[str, Any]]:
        """Get schemas for tables that might be relevant to the query.
        
        Priority:
        1. Tables identified from semantic retrieval (table summaries)
        2. Tables matching query keywords
        3. Fallback to first N tables
        """
        all_tables = self._list_tables()
        relevant_tables = []
        
        # First priority: tables identified from semantic retrieval
        if table_ids_from_semantic:
            for table_id in table_ids_from_semantic:
                bq_table = self._find_bigquery_table_by_table_id(table_id)
                if bq_table and bq_table not in relevant_tables:
                    relevant_tables.append(bq_table)
                    if len(relevant_tables) >= max_tables:
                        break
        
        # Second priority: keyword matching
        if len(relevant_tables) < max_tables:
            query_lower = query.lower()
            key_terms = []
            
            # Common insurance terms
            if "comprehensive" in query_lower:
                key_terms.append("comprehensive")
            if "collision" in query_lower:
                key_terms.append("collision")
            if "hurricane" in query_lower:
                key_terms.append("hurricane")
            if "ho3" in query_lower or "homeowner" in query_lower:
                key_terms.extend(["ho3", "homeowner", "maps"])
            if "rate" in query_lower:
                key_terms.append("rate")
            if "territory" in query_lower or "zip" in query_lower:
                key_terms.extend(["territory", "zip", "disruption"])
            if "motorcycle" in query_lower or "mcy" in query_lower:
                key_terms.extend(["motorcycle", "mcy", "rate"])
            
            for table_id in all_tables:
                if table_id in relevant_tables:
                    continue
                table_lower = table_id.lower()
                if any(term in table_lower for term in key_terms):
                    relevant_tables.append(table_id)
                    if len(relevant_tables) >= max_tables:
                        break
        
        # Fallback: if still not enough, add some tables
        if not relevant_tables:
            relevant_tables = all_tables[:max_tables]
        
        # Get schemas
        schemas = []
        for table_id in relevant_tables:
            schema = self._get_table_schema(table_id)
            if schema.get("columns"):
                schemas.append(schema)
        return schemas

    def _extract_specific_values_from_query(self, query: str) -> List[str]:
        """Extract specific values from the query that can be searched in table content."""
        return extract_specific_values_from_query(query)
        """Extract specific values from the query that can be searched in table content.
        
        Extracts:
        - Zip codes (5-digit numbers)
        - Territory numbers (1-3 digit numbers after "territory")
        - Percentages (e.g., "0.305%", "-0.133%")
        - Model names (capitalized phrases, e.g., "Ducati Panigale V4 R")
        - Specific numbers that might be in tables
        
        Args:
            query: The user's question
            
        Returns:
            List of specific values to search for in table content
        """
        import re
        values = []
        
        # Extract zip codes (5-digit numbers)
        zip_codes = re.findall(r'\b\d{5}\b', query)
        values.extend(zip_codes)
        
        # Extract territory numbers (e.g., "Territory 117", "territory 118")
        # Also look for patterns like "East of Nellis Boulevard (Territory 117)" or just "117" and "118" in context
        territory_matches = re.findall(r'territory\s+(\d{1,3})', query, re.IGNORECASE)
        values.extend(territory_matches)
        
        # Also extract standalone 3-digit numbers that might be territories (117, 118)
        # But only if they appear in context that suggests territories
        if "west" in query.lower() or "east" in query.lower() or "territory" in query.lower():
            potential_territories = re.findall(r'\b(11[0-9]|12[0-9])\b', query)
            values.extend(potential_territories)
        
        # Extract percentages (e.g., "0.305%", "-0.133%")
        percentages = re.findall(r'-?\d+\.?\d*%', query)
        values.extend(percentages)
        
        # Extract specific numbers that might be GRG values or other codes
        # Look for patterns like "GRG 051", "051", "015"
        grg_matches = re.findall(r'(?:grg|rating\s+group)\s+(\d{2,3})', query, re.IGNORECASE)
        values.extend(grg_matches)
        
        # Extract capitalized model names (e.g., "Ducati Panigale V4 R", "Honda Grom ABS")
        # Look for sequences of capitalized words (2-6 words)
        model_names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z0-9]+){1,5})\b', query)
        # Filter out common words
        common_words = {'State', 'Farm', 'Fire', 'Casualty', 'Company', 'Policyholder', 'Resident', 'Living', 'West', 'East', 'Boulevard', 'Proposed', 'Percentage', 'Rate', 'Change', 'Comprehensive', 'Coverage', 'Compared', 'Numeric', 'Difference', 'Between', 'Collision', 'Rating', 'Groups', 'Exceed', 'Points'}
        model_names = [m for m in model_names if not any(word in m for word in common_words)]
        values.extend(model_names)
        
        # Extract capitalized single words that might be brand/model names
        # Look for capitalized words that appear after model-related context
        # This is a heuristic - capitalized words in insurance context often refer to brands/models
        if any(word in query.lower() for word in ['model', 'make', 'brand', 'vehicle', 'motorcycle', 'car', 'truck']):
            # Extract capitalized words that might be brand/model names
            # Look for patterns like "2023 Ducati" or "Honda Grom"
            brand_model_candidates = re.findall(r'\b([A-Z][a-z]{2,})\b', query)
            # Filter out common words and words already captured in model_names
            excluded = {'State', 'Farm', 'Fire', 'Casualty', 'Company', 'Policyholder', 'Resident', 'Living', 
                        'West', 'East', 'Boulevard', 'Proposed', 'Percentage', 'Rate', 'Change', 'Comprehensive', 
                        'Coverage', 'Compared', 'Numeric', 'Difference', 'Between', 'Collision', 'Rating', 
                        'Groups', 'Exceed', 'Points', 'The', 'For', 'And', 'Does', 'Face', 'Higher', 'Than'}
            brand_model_terms = [t for t in brand_model_candidates 
                                if t not in excluded and t.lower() not in [v.lower() for v in values]
                                and len(t) >= 3]  # At least 3 characters
            values.extend(brand_model_terms[:10])  # Limit to avoid noise
        
        # Remove duplicates and empty strings
        values = list(set([v.strip() for v in values if v.strip()]))
        
        return values

    def _sql_retrieval(
        self, 
        query: str, 
        table_ids_from_semantic: List[str] = None,
        calculation_inputs: List[str] = None
    ) -> tuple[str, List[tuple[str, int]]]:
        """Fetch full tables from BigQuery using the retrieval logic module."""
        return self.retrieval_logic.full_table_retrieval(
            query, 
            table_ids_from_semantic=table_ids_from_semantic,
            calculation_inputs=calculation_inputs or []
        )
        """Fetch full tables from BigQuery instead of generating SQL.
        
        This approach:
        - Uses tables identified from semantic search (prioritized)
        - Searches table content for specific values from query (NEW)
        - Fetches full table content (SELECT * LIMIT 200)
        - Converts to Markdown format for better LLM readability
        - Avoids SQL generation issues (column name mismatches, casting errors)
        
        Args:
            query: The user's question (used for fallback keyword matching)
            table_ids_from_semantic: List of table_ids identified from semantic retrieval
        """
        tables = self._list_tables()
        if not tables:
            return "No tables available in the dataset.", []
        
        # Track pages from value-based table matches for later expansion
        value_based_table_pages = []
        
        relevant_tables = []
        
        # Priority 1: Tables identified from semantic search
        if table_ids_from_semantic:
            for table_id in table_ids_from_semantic:
                bq_table = self._find_bigquery_table_by_table_id(table_id)
                if bq_table and bq_table not in relevant_tables:
                    relevant_tables.append(bq_table)
        
        # Debug: Log which tables we're retrieving
        if relevant_tables:
            print(f"\n      → Retrieving {len(relevant_tables)} tables from semantic search", end="", flush=True)
        
        # Priority 2: Search table content for specific values (for queries with specific identifiers)
        # Always run value-based search if query contains specific values (zip codes, territories, percentages, etc.)
        # This ensures we find tables with exact values even if semantic search found many tables
        specific_values = self._extract_specific_values_from_query(query)
        if specific_values and len(specific_values) >= 3:  # Only if we have meaningful specific values
            # For queries with model names or GRG values, search more rows and more tables
            # (models might be deep in large tables)
            has_model_names = any(len(v.split()) > 1 and v[0].isupper() for v in specific_values)
            sample_rows = 50 if has_model_names else 20  # More rows for model searches
            max_search = 300 if has_model_names else 200  # More tables for model searches
            
            # Limit search to most relevant tables (prioritized by name matching)
            value_matches = self._search_tables_by_content(specific_values, sample_rows=sample_rows, max_tables_to_search=max_search, query_context=query)
            
            # Prioritize value-based matches - these contain exact values from the query
            # Move value-based matches to the front, even if they're already in relevant_tables
            value_based_priority = []
            semantic_only = []
            
            # Separate value-based matches from semantic-only matches
            for table_id in relevant_tables:
                if table_id in value_matches:
                    value_based_priority.append(table_id)
                else:
                    semantic_only.append(table_id)
            
            # Add new value-based matches that weren't in semantic results
            # Ensure diversity: include tables matching different types of search values
            # (e.g., zip codes/territories AND model names, not just one type)
            
            # Ensure diversity: when query has multiple types of search values,
            # prioritize tables from different document sources to get comprehensive coverage
            # Group tables by document source (extract document ID from table name)
            tables_by_doc = {}
            for table_id in value_matches:
                if table_id not in value_based_priority:
                    # Extract document identifier (first part of table name after "table__")
                    # e.g., "table__213128717_179157013__..." -> "213128717_179157013"
                    parts = table_id.split('__')
                    if len(parts) >= 2:
                        doc_id = parts[1]  # Document identifier
                        if doc_id not in tables_by_doc:
                            tables_by_doc[doc_id] = []
                        tables_by_doc[doc_id].append(table_id)
            
            # Add tables ensuring we get tables from multiple document sources
            # This ensures diversity (e.g., rate filing docs AND GRG manual docs)
            # Interleave tables from different documents to ensure we get both types
            tables_to_add = []
            max_per_doc = 8  # Limit per document to ensure diversity
            
            # Create iterators for each document
            doc_iterators = {doc_id: iter(doc_tables[:max_per_doc]) for doc_id, doc_tables in tables_by_doc.items()}
            doc_ids_list = list(doc_iterators.keys())
            
            # Round-robin through documents to interleave tables
            doc_index = 0
            while len(tables_to_add) < 20 and doc_ids_list:
                doc_id = doc_ids_list[doc_index % len(doc_ids_list)]
                try:
                    table_id = next(doc_iterators[doc_id])
                    if table_id not in tables_to_add:
                        tables_to_add.append(table_id)
                except StopIteration:
                    # This document is exhausted, remove it
                    doc_ids_list = [d for d in doc_ids_list if d != doc_id]
                    if not doc_ids_list:
                        break
                    continue
                
                doc_index += 1
                
                # Safety: prevent infinite loop
                if doc_index > 100:
                    break
            
            # Add to priority list
            for table_id in tables_to_add:
                if table_id not in value_based_priority:
                    value_based_priority.append(table_id)
                    if len(value_based_priority) >= 22:  # Slightly higher to ensure diversity
                        break
            
            # If we still have room, add remaining value-based matches
            if len(value_based_priority) < 22:
                for table_id in value_matches:
                    if table_id not in value_based_priority:
                        value_based_priority.append(table_id)
                        if len(value_based_priority) >= 22:
                            break
            
            # Document-level expansion: If we found tables from a document via value-based search,
            # also include other tables from that same document (up to a reasonable limit)
            # This ensures we get related tables (e.g., territory definitions) even if they
            # don't contain the exact search values in their names or first N rows
            expanded_doc_tables = set(value_based_priority)  # Track all tables we're adding
            for table_id in value_based_priority[:10]:  # For top 10 value-based matches
                # Extract document identifier
                parts = table_id.split('__')
                if len(parts) >= 2:
                    doc_id = parts[1]
                    # Find all tables from this document
                    doc_tables = [t for t in tables if f'__{doc_id}__' in t]
                    # Add up to 5 additional tables from same document (if not already included)
                    added = 0
                    for doc_table in doc_tables:
                        if doc_table not in expanded_doc_tables and doc_table not in semantic_only:
                            value_based_priority.append(doc_table)
                            expanded_doc_tables.add(doc_table)
                            added += 1
                            if added >= 5:  # Limit to avoid too many tables from one doc
                                break
            
            # Combine: value-based first, then semantic-only
            # If we have value-based matches from multiple document sources, use slightly more tables
            # to ensure comprehensive coverage (e.g., both rate filing AND GRG manual tables)
            num_doc_sources = len(tables_by_doc) if 'tables_by_doc' in locals() else 1
            # Use 22 tables if we have multiple document sources to ensure we get all critical tables
            max_tables = 22 if num_doc_sources >= 2 and len(value_based_priority) >= 15 else 20
            
            relevant_tables = value_based_priority[:max_tables]
            remaining = max_tables - len(relevant_tables)
            if remaining > 0:
                relevant_tables.extend(semantic_only[:remaining])
            
            # Final limit: 22 tables max (slightly higher for multi-document queries)
            # This ensures we get comprehensive coverage when query needs data from multiple sources
            if len(relevant_tables) > 22:
                relevant_tables = value_based_priority[:22]
            
            if value_based_priority:
                num_new = len(value_based_priority) - len([t for t in value_based_priority if t in (table_ids_from_semantic or [])])
                print(f"\n      → Prioritized {len(value_based_priority)} value-based tables ({num_new} new, {len(specific_values)} values: {', '.join(specific_values[:5])})", end="", flush=True)
        
        # Priority 3: Search by column names for specific data types (more targeted)
        if len(relevant_tables) < 20:
            column_keywords = []
            query_lower = query.lower()
            
            # Extract column-related keywords from query (more specific)
            if "base rate" in query_lower:
                column_keywords.extend(["base_rate", "ho3_a_rate", "coverage_a"])
            elif "rate" in query_lower and "premium" not in query_lower:
                column_keywords.extend(["rate", "rating"])
            
            if "territory" in query_lower:
                column_keywords.extend(["territory", "zip_code"])
            elif "zip" in query_lower:
                column_keywords.extend(["zip", "zip_code"])
            
            if "comprehensive" in query_lower:
                column_keywords.extend(["comprehensive"])
            
            if "grg" in query_lower or ("collision" in query_lower and "rating" in query_lower):
                column_keywords.extend(["grg", "collision_rating_group"])
            
            if "hurricane" in query_lower and "deductible" in query_lower:
                column_keywords.extend(["hurricane_deductible", "mandatory_hurricane"])
            elif "hurricane" in query_lower:
                column_keywords.extend(["hurricane"])
            
            if "motorcycle" in query_lower or "mcy" in query_lower:
                column_keywords.extend(["motorcycle", "mcy", "model"])
            
            # Only search if we have specific keywords (not generic "rate")
            if column_keywords and len(relevant_tables) < 12:
                column_matches = self._search_tables_by_column_name(column_keywords)
                
                # Filter matches: prioritize tables with multiple matching columns
                scored_matches = []
                for table_id in column_matches:
                    if table_id in relevant_tables:
                        continue
                    try:
                        schema = self._get_table_schema(table_id)
                        columns = [col['name'].lower() for col in schema.get('columns', [])]
                        # Score: number of matching keywords found in columns
                        score = sum(1 for kw in column_keywords if any(kw.lower() in col for col in columns))
                        if score > 0:
                            scored_matches.append((table_id, score))
                    except Exception:
                        continue
                
                # Sort by score and take top matches
                scored_matches.sort(key=lambda x: x[1], reverse=True)
                for table_id, score in scored_matches[:20 - len(relevant_tables)]:
                    relevant_tables.append(table_id)
                
                if scored_matches and len(relevant_tables) > len(table_ids_from_semantic or []):
                    added = len(relevant_tables) - len(table_ids_from_semantic or [])
                    print(f"\n      → Found {added} additional tables via column search (from {len(column_matches)} candidates)", end="", flush=True)
        
        # Priority 4: Fallback to keyword matching (limit to 20 to get more coverage)
        if len(relevant_tables) < 20:
            # Use keyword matching to find relevant tables
            query_lower = query.lower()
            key_terms = []
            
            if "comprehensive" in query_lower:
                key_terms.append("comprehensive")
            if "collision" in query_lower:
                key_terms.append("collision")
            if "hurricane" in query_lower:
                key_terms.append("hurricane")
            if "ho3" in query_lower or "homeowner" in query_lower:
                key_terms.extend(["ho3", "homeowner", "maps"])
            if "rate" in query_lower:
                key_terms.append("rate")
            if "territory" in query_lower or "zip" in query_lower:
                key_terms.extend(["territory", "zip", "disruption"])
            if "motorcycle" in query_lower or "mcy" in query_lower:
                key_terms.extend(["motorcycle", "mcy", "rate"])
            
            for table_id in tables:
                table_lower = table_id.lower()
                if any(term in table_lower for term in key_terms):
                    relevant_tables.append(table_id)
                    if len(relevant_tables) >= 20:  # Limit set to 20 tables
                        break
            
            if relevant_tables and len(relevant_tables) > len(table_ids_from_semantic or []):
                print(f"\n      → Found {len(relevant_tables) - len(table_ids_from_semantic or [])} additional tables via keyword matching", end="", flush=True)
        
        if not relevant_tables:
            return "No relevant tables found for this query."
        
        # Limit to top 22 tables (slightly higher for multi-document queries)
        # This ensures comprehensive coverage when query needs data from multiple document sources
        max_final_tables = 22
        relevant_tables = relevant_tables[:max_final_tables]
        
        if len(relevant_tables) > max_final_tables:
            print(f"\n      → Limiting to top {max_final_tables} tables (found {len(relevant_tables)} total)", end="", flush=True)
        
        full_tables_content = []
        
        # Detect and group split tables (same table split across pages)
        print(f"    - Analyzing {len(relevant_tables)} tables for split detection...", end=" ", flush=True)
        grouped_tables = self._group_split_tables(relevant_tables[:max_final_tables])
        
        num_groups = sum(1 for g in grouped_tables if isinstance(g, list))
        num_single = sum(1 for g in grouped_tables if not isinstance(g, list))
        if num_groups > 0:
            print(f"\n      → Found {num_groups} split table groups, {num_single} individual tables", end="", flush=True)
        
        print(f"\n    - Fetching {len(grouped_tables)} tables/groups...", end=" ", flush=True)
        
        # Inspect tables first to verify they have relevant data
        table_inspections = {}
        for table_group in grouped_tables:
            # Inspect first table in group to get structure
            first_table = table_group[0] if isinstance(table_group, list) else table_group
            inspection = self._inspect_table_content(first_table, max_rows=3)
            if isinstance(table_group, list):
                # Group of split tables
                for table_id in table_group:
                    table_inspections[table_id] = inspection
            else:
                # Single table
                table_inspections[table_group] = inspection
        
        # Track pages from value-based tables for text expansion
        # Get page numbers from table metadata
        for table_id in relevant_tables[:20]:  # For top 20 tables
            try:
                # Query to get page number from table metadata
                page_query = f"SELECT DISTINCT meta_page_number, meta_source_pdf FROM `{self.project_id}.{self.dataset_name}.{table_id}` WHERE meta_page_number IS NOT NULL LIMIT 1"
                page_job = self.bq_client.query(page_query)
                page_rows = list(page_job.result())
                if page_rows:
                    page_num = page_rows[0].get('meta_page_number')
                    source = page_rows[0].get('meta_source_pdf')
                    if page_num is not None and source:
                        value_based_table_pages.append((source, int(page_num)))
            except Exception:
                continue
        
        # Process tables (groups or individual)
        # Split tables need sequential processing, but individual tables can be parallelized
        full_tables_content = []
        table_idx = 0
        
        # First, process split table groups sequentially (they need to be merged)
        split_groups = []
        single_tables = []
        for table_group in grouped_tables:
            if isinstance(table_group, list):
                split_groups.append(table_group)
            else:
                single_tables.append(table_group)
        
        # Process split groups sequentially
        for table_group in split_groups:
            table_idx += 1
            merged_content = self._merge_split_tables(table_group, table_idx, len(grouped_tables))
            if merged_content:
                full_tables_content.append((table_idx, merged_content))
        
        # Process individual tables in parallel
        if single_tables:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {}
                start_idx = len(split_groups) + 1
                for idx, table_id in enumerate(single_tables, start=start_idx):
                    future = executor.submit(self._fetch_single_table_parallel, table_id, idx, len(grouped_tables))
                    futures[future] = idx
                
                # Collect results as they complete
                results = {}
                for future in as_completed(futures):
                    idx, content = future.result()
                    if content:
                        results[idx] = content
                
                # Add results in order
                for idx in sorted(results.keys()):
                    full_tables_content.append((idx, results[idx]))
        
        # Sort by index and extract content
        full_tables_content.sort(key=lambda x: x[0])
        final_content = [content for _, content in full_tables_content]
        
        if not final_content:
            print("(no data)")
            return "No table data retrieved.", []
        
        print(f"({len(final_content)} tables/groups retrieved)")
        
        # Return both table content and pages for expansion
        return "\n\n".join(final_content), value_based_table_pages
    
    # Note: _group_split_tables, _are_likely_splits, _merge_split_tables, 
    # _fetch_single_table, and _fetch_single_table_parallel have been moved to retrieval_logic.py
    # They are now accessed via self.retrieval_logic
    
    # Note: The following methods have been moved to retrieval_logic.py:
    # - _group_split_tables
    # - _are_likely_splits  
    # - _merge_split_tables
    # - _fetch_single_table
    # - _fetch_single_table_parallel
    # They are now accessed via self.retrieval_logic.full_table_retrieval()
    
    def _is_calculation_query(self, question: str) -> bool:
        """Detect if a question requires calculation.
        
        Args:
            question: The user's question
            
        Returns:
            True if the question likely requires calculation
        """
        question_lower = question.lower()
        calculation_keywords = [
            'calculate', 'compute', 'determine', 'find', 'what is',
            'premium', 'rate', 'factor', 'multiply', 'multiplier',
            'deductible', 'discount', 'surcharge', 'adjustment'
        ]
        return any(keyword in question_lower for keyword in calculation_keywords)
    
    def _extract_calculation_inputs(self, question: str) -> List[str]:
        """Extract inputs needed for calculation from the question.
        
        Args:
            question: The user's question
            
        Returns:
            List of input terms to search for in tables
        """
        question_lower = question.lower()
        inputs = []
        
        # Extract policy/coverage types
        if 'ho3' in question_lower or 'ho-3' in question_lower:
            inputs.extend(['ho3', 'ho-3', 'homeowner', 'homeowner 3'])
        if 'coverage a' in question_lower:
            inputs.append('coverage a')
        
        # Extract amounts/limits
        import re
        amounts = re.findall(r'\$[\d,]+', question)
        inputs.extend([a.replace('$', '').replace(',', '') for a in amounts])
        
        # Extract distance/location info
        distances = re.findall(r'(\d+[,.]?\d*)\s*(?:feet|ft|miles|mi)', question_lower)
        inputs.extend(distances)
        
        # Extract calculation-related terms
        if 'base rate' in question_lower:
            inputs.extend(['base rate', 'rate', 'base', 'hurricane base rate', 'hurricane rate'])
        if 'deductible factor' in question_lower or 'deductible' in question_lower:
            inputs.extend(['deductible factor', 'deductible', 'factor'])
        if 'hurricane' in question_lower:
            inputs.extend(['hurricane', 'hurricane deductible', 'mandatory hurricane', 'hurricane base rate'])
        if 'coast' in question_lower or 'coastline' in question_lower:
            inputs.extend(['coast', 'coastline', 'distance to coast'])
        
        # For HO3 base rate questions, add document-specific search terms
        if 'ho3' in question_lower and 'base rate' in question_lower:
            inputs.extend(['maps rate pages', 'rate pages', 'exhibit', '215004905'])
        
        return list(set(inputs))  # Remove duplicates

    # ----- Public API ------------------------------------------------------------

    def answer(self, question: str) -> Dict[str, Any]:
        """Full RAG flow for a single question.
        
        Always uses hybrid approach: both semantic and quantitative retrieval.
        """
        contexts: List[str] = []
        source_meta: List[Dict[str, Any]] = []

        # Step 1: Semantic retrieval (always performed)
        print("    - Performing semantic retrieval from vector store...", end=" ", flush=True)
        # For "list all" queries, retrieve and use more results
        is_list_query = "list all" in question.lower() or ("list" in question.lower() and "all" in question.lower())
        k_retrieve = 40 if is_list_query else 25  # Increased for comprehensive list queries
        top_k_use = 20 if is_list_query else 12  # Increased for comprehensive list queries
        
        sem = self._semantic_retrieval(question, k=k_retrieve)
        print(f"✓ (retrieved {len(sem)} candidates)")
        print("    - Reranking results...", end=" ", flush=True)
        top_sem = sem[:top_k_use]
        print(f"✓ (selected top {len(top_sem)})")
        
        # For geographic/territory queries, perform targeted search for street names and geographic boundaries
        # This helps find territory definitions that use street names (e.g., "West of Nellis Boulevard")
        import re
        geographic_patterns = [
            r'(?:west|east|north|south)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:boulevard|street|avenue|road|drive|way)\s+([A-Z][a-z]+)',
        ]
        geographic_terms = []
        for pattern in geographic_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            geographic_terms.extend(matches)
        
        if geographic_terms:
            print("    - Performing targeted search for geographic boundaries...", end=" ", flush=True)
            geo_results = []
            for term in geographic_terms[:3]:  # Limit to top 3 terms
                try:
                    # Search for the geographic term combined with relevant context
                    search_query = f"{term} ZIP Code territory"
                    search_results = self._semantic_retrieval(search_query, k=5)
                    for result in search_results:
                        # Avoid duplicates
                        content_hash = hash(result.get("content", ""))
                        if not any(hash(r.get("content", "")) == content_hash for r in top_sem):
                            geo_results.append(result)
                except Exception:
                    continue
            
            if geo_results:
                top_sem.extend(geo_results)
                print(f"✓ (found {len(geo_results)} geographic boundary items)")
            else:
                print("✓ (no geographic items found)")
        
        # For "list all" queries, perform additional targeted searches to ensure completeness
        # Extract key terms from initial results and query to find related items
        if is_list_query:
            print("    - Performing additional targeted searches for comprehensive coverage...", end=" ", flush=True)
            
            # Strategy 1: Search for table of contents, index, or summary pages
            toc_queries = [
                "table of contents",
                "index",
                "summary",
                "list of rules",
                "all rules",
                "rating plan rules"
            ]
            toc_results = []
            for toc_query in toc_queries:
                try:
                    search_results = self._semantic_retrieval(toc_query, k=5)
                    for result in search_results:
                        content_hash = hash(result.get("content", ""))
                        if not any(hash(r.get("content", "")) == content_hash for r in top_sem):
                            toc_results.append(result)
                except Exception:
                    continue
            
            if toc_results:
                top_sem.extend(toc_results[:10])  # Add top 10 TOC results
                print(f"\n      → Found {len(toc_results)} table of contents/index items", end="", flush=True)
            
            # Strategy 2: Extract key terms/phrases from the initial results
            # Look for bullet points, list items, rule names, etc.
            key_terms = []
            seen_terms = set()
            
            # Extract capitalized phrases and rule-like patterns from top results
            import re
            for result in top_sem[:15]:  # Analyze top 15 results (increased from 10)
                content = result.get("content", "")
                # Find numbered items (e.g., "Rule C-1", "Rule C-2", "Factor 1", etc.)
                numbered_items = re.findall(r'\b(?:rule|factor|discount|item)\s+[a-z]?[-]?\d+[:\s]+([^,\n*]+?)(?:,|\n|$)', content, re.IGNORECASE)
                key_terms.extend([item.strip() for item in numbered_items if len(item.strip()) > 5])
                
                # Find capitalized phrases (potential rule/item names)
                phrases = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', content)
                for phrase in phrases:
                    # Filter for meaningful phrases (2+ words, not too common)
                    words = phrase.split()
                    if len(words) >= 2 and len(phrase) > 10:
                        phrase_lower = phrase.lower()
                        # Skip very common words
                        if phrase_lower not in seen_terms and not any(
                            word.lower() in ['the', 'and', 'for', 'with', 'from', 'this', 'that'] 
                            for word in words[:2]
                        ):
                            key_terms.append(phrase)
                            seen_terms.add(phrase_lower)
            
            # Also extract key terms from the query itself
            query_words = question.split()
            # Find capitalized terms in query
            for i, word in enumerate(query_words):
                if word and word[0].isupper() and len(word) > 3:
                    if i + 1 < len(query_words) and query_words[i+1] and query_words[i+1][0].isupper():
                        # Two-word capitalized phrase
                        phrase = f"{word} {query_words[i+1]}"
                        if phrase.lower() not in seen_terms:
                            key_terms.append(phrase)
                            seen_terms.add(phrase.lower())
            
            # Limit to top 20 most relevant terms (increased from 15)
            key_terms = key_terms[:20]
            
            additional_results = []
            if key_terms:
                for search_term in key_terms:
                    try:
                        search_results = self._semantic_retrieval(search_term, k=3)  # Increased from 2
                        for result in search_results[:2]:  # Top 2 results per search (increased from 1)
                            # Avoid duplicates by content hash
                            content_hash = hash(result.get("content", ""))
                            if not any(hash(r.get("content", "")) == content_hash for r in top_sem):
                                additional_results.append(result)
                    except Exception:
                        continue
            
            if additional_results:
                top_sem.extend(additional_results)
                print(f"\n      → Found {len(additional_results)} additional items from {len(key_terms)} key terms", end="", flush=True)
            else:
                print("✓ (no additional items found)")
        
        # Expand to include all content from same pages and documents
        print("    - Expanding to include all content from same pages/documents...", end=" ", flush=True)
        expanded_sem = self._expand_by_page_and_document(top_sem, max_expansion=300)  # Increased expansion limit
        original_count = len(top_sem)
        expanded_count = len(expanded_sem)
        if expanded_count > original_count:
            print(f"✓ (expanded from {original_count} to {expanded_count} items)")
        else:
            print(f"✓ (no expansion needed)")
        
        # Extract table IDs from table summaries found in semantic retrieval (from expanded results)
        table_ids_from_semantic = self._extract_table_ids_from_results(expanded_sem)
        if table_ids_from_semantic:
            print(f"    - Found {len(table_ids_from_semantic)} relevant tables from semantic search")
        
        # Use expanded results instead of top_sem
        top_sem = expanded_sem
        
        # Add semantic results to context
        for r in top_sem:
            meta = r["metadata"]
            element_type = meta.get("element_type", "text")
            is_expanded = r.get("expanded", False)
            
            # Build header with all relevant info
            source = meta.get('source', 'unknown')
            page_num = meta.get('page_number', '?')
            header = f"[SOURCE: {source}, page {page_num}"
            if element_type == "table_summary":
                table_id = meta.get('table_id', 'unknown')
                header += f", TABLE: {table_id}"
            if is_expanded:
                header += ", EXPANDED"
            header += "]"
            
            contexts.append(header + "\n" + r["content"])
            source_meta.append(meta)

        # Step 2: Full table retrieval (always performed, using tables identified from semantic search)
        print("    - Retrieving full tables from BigQuery...", end=" ", flush=True)
        
        # For calculation queries, enhance retrieval with calculation-specific inputs
        calculation_inputs = []
        if self._is_calculation_query(question):
            calculation_inputs = self._extract_calculation_inputs(question)
            if calculation_inputs:
                print(f"\n      → Detected calculation query, searching for: {', '.join(calculation_inputs[:5])}", end="", flush=True)
        
        sql_result, value_based_table_pages = self._sql_retrieval(
            question, 
            table_ids_from_semantic=table_ids_from_semantic,
            calculation_inputs=calculation_inputs
        )
        # Note: _sql_retrieval now prints its own progress
        contexts.append("Table data:\n" + sql_result)
        source_meta.append({"type": "sql", "content": sql_result})
        
        # Step 2b: Expand to include text chunks from pages of value-based table matches
        # This ensures we get territory definitions and other text that explains table data
        if value_based_table_pages:
            print("    - Expanding to include text from pages with value-based tables...", end=" ", flush=True)
            page_expansion_results = []
            seen_content = {r["content"] for r in top_sem}
            
            for source, page_num in value_based_table_pages[:10]:  # Limit to top 10 pages
                try:
                    sql = f"""
                    SELECT 
                        content,
                        source,
                        page_number,
                        element_type,
                        table_id,
                        chunk_id
                    FROM `{self.project_id}.{self.dataset_name}.{self.table_name}`
                    WHERE source = @source
                      AND page_number = @page_num
                      AND element_type != 'table_summary'
                    LIMIT 50
                    """
                    
                    job_config = bigquery.QueryJobConfig(
                        query_parameters=[
                            bigquery.ScalarQueryParameter("source", "STRING", source),
                            bigquery.ScalarQueryParameter("page_num", "INT64", int(page_num)),
                        ]
                    )
                    
                    job = self.bq_client.query(sql, job_config=job_config)
                    rows = list(job.result())
                    
                    for row in rows:
                        doc_content = row.get("content", "")
                        if not doc_content or doc_content in seen_content:
                            continue
                        
                        doc_meta = {
                            "source": row.get("source"),
                            "page_number": row.get("page_number"),
                            "element_type": row.get("element_type", "text"),
                        }
                        
                        if row.get("table_id"):
                            doc_meta["table_id"] = row.get("table_id")
                        
                        page_expansion_results.append({
                            "content": doc_content,
                            "score": 0.7,
                            "metadata": doc_meta,
                            "expanded": True,
                        })
                        seen_content.add(doc_content)
                        
                        if len(page_expansion_results) >= 100:  # Limit expansion
                            break
                except Exception:
                    continue
            
            if page_expansion_results:
                top_sem.extend(page_expansion_results)
                # Add to contexts
                for r in page_expansion_results:
                    meta = r["metadata"]
                    source = meta.get('source', 'unknown')
                    page_num = meta.get('page_number', '?')
                    header = f"[SOURCE: {source}, page {page_num}, EXPANDED FROM TABLE PAGE]"
                    contexts.append(header + "\n" + r["content"])
                    source_meta.append(meta)
                
                print(f"✓ (added {len(page_expansion_results)} text chunks from {len(value_based_table_pages)} pages)")
            else:
                print("✓ (no additional text found)")

        context_block = "\n\n".join(contexts) if contexts else "No context."
        
        # Monitor input token usage (rough estimate: 1 token ≈ 4 characters)
        context_length = len(context_block)
        prompt_length = len(question)
        estimated_input_tokens = (context_length + prompt_length) / 4
        
        # Gemini 2.5 Pro has 128K token context window
        # Warn if we're approaching the limit (though we should be well under)
        if estimated_input_tokens > 100000:
            print(f"⚠ (input tokens: ~{estimated_input_tokens:.0f} - approaching limit)")
        elif estimated_input_tokens > 50000:
            print(f"    (input tokens: ~{estimated_input_tokens:.0f})", end=" ", flush=True)

        print("    - Generating answer with LLM...", end=" ", flush=True)
        
        # Enhanced prompt for list queries
        list_instruction = ""
        if is_list_query:
            list_instruction = (
                "\nIMPORTANT: This is a 'list all' query. Be comprehensive and include ALL items mentioned in the context. "
                "If the context contains table summaries or structured data, extract all relevant items from both semantic text and table data. "
                "Look for table of contents, index pages, or summary tables that may list all items. "
                "If items are numbered (e.g., Rule C-1, Rule C-2, etc.), ensure you capture all numbered items in sequence.\n"
            )
        
        # Enhanced prompt for calculation queries
        calculation_instruction = ""
        is_calculation_query = self._is_calculation_query(question)
        if is_calculation_query:
            calculation_instruction = (
                "\nIMPORTANT: This is a calculation question. "
                "You need to identify all required inputs (rates, factors, amounts, etc.) and find them in the provided tables. "
                "Search through ALL tables systematically to locate the necessary values. "
                "Once you have all required values, perform the calculation and provide the result. "
                "If you cannot find a required value after thoroughly searching all provided tables, explicitly state which value is missing.\n"
            )
        
        # Format prompt template with dynamic values
        prompt = self.prompt_template.format(
            list_instruction=list_instruction,
            calculation_instruction=calculation_instruction,
            context_block=context_block,
            question=question
        )

        answer_text = self.llm.invoke(prompt)
        
        # Convert to string for truncation detection
        answer_str = str(answer_text) if not isinstance(answer_text, str) else answer_text
        answer_length = len(answer_str)
        
        # Log response length for debugging
        # Rough estimate: 1 token ≈ 4 characters, so 8192 tokens ≈ 32,768 chars
        # We'll warn if response is very long (might indicate truncation risk)
        if answer_length > 30000:  # Very long response
            print(f"⚠ (response length: {answer_length} chars)")
        else:
            print("✓")
        
        # Check for common truncation patterns (though Gemini usually doesn't truncate mid-sentence)
        # Store the string version for return
        answer_text = answer_str

        return {
            "answer": answer_text,
            "sources": source_meta,
            "query_category": "hybrid",  # Always hybrid since we use both semantic and SQL
        }


