from __future__ import annotations

"""
DOC Index: vector store backed by BigQuery (or any LangChain-compatible store).

Responsibilities:
- Turn parsed text elements and table summaries into LangChain Documents
- Build a vector index using Vertex AI embeddings + BigQueryVectorStore
"""

from typing import List, Dict, Any, TYPE_CHECKING
import warnings

from langchain_core.documents import Document

from src.parsing.pdf_parser import ParsedDocument, ParsedChart
from src.parsing.table_processor import ProcessedTable

if TYPE_CHECKING:
    from langchain_google_community import BigQueryVectorStore

# Suppress deprecation warnings for VertexAIEmbeddings from langchain-google-vertexai.
# These wrappers still work; long-term we can migrate to langchain-google-genai embeddings.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="langchain_google_vertexai",
)

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_google_vertexai import VertexAIEmbeddings
    from langchain_google_community import BigQueryVectorStore
except Exception:  # pragma: no cover - optional deps
    RecursiveCharacterTextSplitter = None  # type: ignore
    VertexAIEmbeddings = None  # type: ignore
    BigQueryVectorStore = None  # type: ignore


class VectorIndexBuilder:
    """Builds the DOC Index (vector store) from parsed documents and processed tables."""

    def __init__(self, project_id: str, dataset_name: str = "rag_dataset", location: str = "US") -> None:
        if (
            RecursiveCharacterTextSplitter is None
            or VertexAIEmbeddings is None
            or BigQueryVectorStore is None
        ):
            raise ImportError(
                "langchain, langchain-google-vertexai, and langchain-google-community "
                "are required for VectorIndexBuilder."
            )

        self.project_id = project_id
        self.dataset_name = dataset_name
        self.location = location

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.embeddings = VertexAIEmbeddings(model_name="text-embedding-005")

    def _add_documents_batched(
        self,
        store: "BigQueryVectorStore",
        docs: List[Document],
        max_instances: int = 200,
        max_chars: int = 60000,
    ) -> None:
        """Add documents to the store in batches that respect Vertex AI limits.

        Vertex text-embedding-005 has:
        - Max ~250 instances per prediction
        - Max ~20k tokens per request (we approximate via character count)
        """
        if not docs:
            return

        batch: List[Document] = []
        batch_chars = 0

        for doc in docs:
            text = doc.page_content or ""
            length = len(text)

            # If adding this doc would exceed instance or char budget, flush current batch
            if batch and (
                len(batch) + 1 > max_instances or batch_chars + length > max_chars
            ):
                store.add_documents(batch)
                batch = []
                batch_chars = 0

            # If a single document is longer than max_chars, send it alone
            if length > max_chars and not batch:
                store.add_documents([doc])
                continue

            batch.append(doc)
            batch_chars += length

        if batch:
            store.add_documents(batch)

    def _docs_from_text(self, parsed_docs: List[ParsedDocument]) -> List[Document]:
        docs: List[Document] = []

        for doc in parsed_docs:
            for el in doc.text_elements:
                chunks = self.text_splitter.split_text(el.text)
                for idx, chunk in enumerate(chunks):
                    docs.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": doc.pdf_path,
                                "page_number": el.page_number,
                                "element_type": el.type,
                                "chunk_id": idx,
                            },
                        )
                    )
        return docs

    def _docs_from_tables(self, tables: List[ProcessedTable]) -> List[Document]:
        docs: List[Document] = []
        for t in tables:
            docs.append(
                Document(
                    page_content=t.summary,
                    metadata={
                        "source": t.pdf_path,
                        "page_number": t.page_number,
                        "element_type": "table_summary",
                        "table_id": t.table_id,
                    },
                )
            )
        return docs

    def _docs_from_charts(self, charts: List[ParsedChart]) -> List[Document]:
        """Create documents from chart data for semantic search."""
        docs: List[Document] = []
        for chart in charts:
            # Create a text representation of the chart for embedding
            chart_text_parts = []
            if chart.title:
                chart_text_parts.append(f"Chart title: {chart.title}")
            if chart.chart_type:
                chart_text_parts.append(f"Chart type: {chart.chart_type}")
            if chart.data:
                # Convert chart data to a readable format
                if isinstance(chart.data, dict):
                    data_str = ", ".join(f"{k}: {v}" for k, v in list(chart.data.items())[:10])
                    if data_str:
                        chart_text_parts.append(f"Chart data: {data_str}")
            
            chart_text = " | ".join(chart_text_parts) if chart_text_parts else "Chart visualization"
            
            docs.append(
                Document(
                    page_content=chart_text,
                    metadata={
                        "source": chart.metadata.get("source", ""),
                        "page_number": chart.page_number,
                        "element_type": "chart",
                        "chart_type": chart.chart_type or "unknown",
                    },
                )
            )
        return docs

    def get_or_create_store(self, table_name: str = "doc_index"):
        """Get or create the vector store (for incremental additions)."""
        store = BigQueryVectorStore(
            project_id=self.project_id,
            dataset_name=self.dataset_name,
            table_name=table_name,
            location=self.location,
            embedding=self.embeddings,
        )
        return store

    def add_document_to_index(
        self,
        parsed_doc: ParsedDocument,
        processed_tables: List[ProcessedTable],
        store: "BigQueryVectorStore",
    ) -> None:
        """Add a single document's content to the vector index incrementally."""
        # Extract text chunks from this document
        text_docs = self._docs_from_text([parsed_doc])
        
        # Extract table summaries from this document's tables
        table_docs = self._docs_from_tables(processed_tables)
        
        # Extract chart summaries from this document's charts
        chart_docs = self._docs_from_charts(parsed_doc.charts)
        
        all_docs = text_docs + table_docs + chart_docs
        if all_docs:
            # Batch by both instance count and approximate token budget
            self._add_documents_batched(store, all_docs)
            print(
                f"    ✓ Added {len(all_docs)} documents to vector index "
                f"({len(text_docs)} text chunks, {len(table_docs)} table summaries, {len(chart_docs)} charts)"
            )

    def build_index(
        self,
        parsed_docs: List[ParsedDocument],
        processed_tables: List[ProcessedTable],
        table_name: str = "doc_index",
    ):
        """Create or update the DOC Index in BigQuery (batch mode)."""
        print("  - Extracting text chunks from documents...")
        text_docs = self._docs_from_text(parsed_docs)
        print(f"    ✓ Created {len(text_docs)} text chunk documents")
        
        print("  - Creating documents from table summaries...")
        table_docs = self._docs_from_tables(processed_tables)
        print(f"    ✓ Created {len(table_docs)} table summary documents")
        
        all_docs = text_docs + table_docs
        print(f"  - Total documents to index: {len(all_docs)}")
        print("  - Generating embeddings and uploading to BigQuery (this may take a while)...")
        
        store = BigQueryVectorStore(
            project_id=self.project_id,
            dataset_name=self.dataset_name,
            table_name=table_name,
            location=self.location,
            embedding=self.embeddings,
        )

        # Batch documents to respect Vertex embeddings instance and token limits
        if all_docs:
            self._add_documents_batched(store, all_docs)
        print(f"  ✓ Successfully indexed {len(all_docs)} documents in BigQuery vector store")
        return store


