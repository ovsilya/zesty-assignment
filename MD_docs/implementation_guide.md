# Implementation Guide: Unified SOTA RAG Architecture

## Quick Start: Prototype Implementation

This guide provides step-by-step instructions to implement the unified architecture for the take-home assignment.

---

## Prerequisites

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install unstructured[all-docs] langchain langchain-google-vertexai langchain-google-community
pip install llama-index llama-parse  # Alternative parser option
pip install pandas pyarrow google-cloud-bigquery
pip install google-cloud-aiplatform
pip install flashrank  # Or: pip install cohere for Cohere rerank
pip install ragas datasets
pip install python-dotenv
```

### 2. GCP Setup (Recommended)

```bash
# Install GCP CLI and authenticate
gcloud auth login
gcloud auth application-default login

# Set up service account (optional, for production)
# Create service account in GCP Console
# Download JSON key file
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
```

### 3. Environment Variables

Create `.env` file:
```bash
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
# Optional: If using OpenAI instead
OPENAI_API_KEY=your-openai-key
```

---

## Phase 1: Document Processing Pipeline

### Step 1.1: PDF Parsing Script

Create `src/parsing/pdf_parser.py`:

```python
import os
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
import pandas as pd
import json
from typing import List, Dict, Any

class PDFParser:
    def __init__(self, output_dir: str = "processed_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Parse a single PDF and extract text and tables."""
        print(f"Parsing {pdf_path}...")
        
        # Use hi_res strategy for better table detection
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=False,
        )
        
        # Separate text and tables
        text_elements = []
        tables = []
        
        for element in elements:
            if hasattr(element, 'metadata'):
                element.metadata.source_file = pdf_path
                
            if element.category == "Table":
                # Convert table to DataFrame
                try:
                    df = pd.DataFrame(element.metadata.text_as_html)
                    # Or use element.metadata.text_as_html and parse with pd.read_html
                    html_str = element.metadata.text_as_html if hasattr(element.metadata, 'text_as_html') else str(element)
                    tables.append({
                        "element": element,
                        "html": html_str,
                        "page_number": element.metadata.page_number if hasattr(element.metadata, 'page_number') else None,
                    })
                except Exception as e:
                    print(f"Error processing table: {e}")
                    # Fallback: store as text
                    text_elements.append(element)
            else:
                text_elements.append(element)
        
        return {
            "text_elements": text_elements,
            "tables": tables,
            "pdf_path": pdf_path,
        }
    
    def process_all_pdfs(self, pdf_dir: str) -> List[Dict[str, Any]]:
        """Process all PDFs in a directory."""
        pdf_dir = Path(pdf_dir)
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        results = []
        for pdf_file in pdf_files:
            try:
                parsed = self.parse_pdf(str(pdf_file))
                results.append(parsed)
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                continue
        
        return results

if __name__ == "__main__":
    parser = PDFParser()
    # Process PDFs from artifacts folders
    results = parser.process_all_pdfs("artifacts/1")
    results.extend(parser.process_all_pdfs("artifacts/2"))
    
    # Save intermediate results
    import pickle
    with open("processed_data/parsed_pdfs.pkl", "wb") as f:
        pickle.dump(results, f)
```

### Step 1.2: Table Extraction and Normalization

Create `src/parsing/table_processor.py`:

```python
import pandas as pd
from langchain_google_vertexai import VertexAI
from langchain.prompts import PromptTemplate
from typing import List, Dict
import re

class TableProcessor:
    def __init__(self):
        self.llm = VertexAI(model_name="gemini-1.5-pro")
        
    def extract_table_to_dataframe(self, table_html: str) -> pd.DataFrame:
        """Convert HTML table to DataFrame."""
        try:
            # Parse HTML table
            dfs = pd.read_html(table_html)
            if dfs:
                return dfs[0]
        except:
            pass
        
        # Fallback: try to parse from text
        # This is a simplified version - you may need more robust parsing
        return pd.DataFrame()
    
    def normalize_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize table data."""
        # Remove empty rows/columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Remove duplicate headers
        if len(df) > 0:
            first_row = df.iloc[0]
            if all(str(val).lower() in ['nan', 'none', ''] for val in first_row):
                df = df.iloc[1:].reset_index(drop=True)
        
        return df
    
    def generate_table_summary(self, df: pd.DataFrame, context: str = "") -> str:
        """Generate natural language summary of table using LLM."""
        # Convert DataFrame to markdown for LLM
        table_md = df.to_markdown(index=False)
        
        prompt = PromptTemplate(
            input_variables=["table", "context"],
            template="""
            Analyze this table and provide a concise summary that captures:
            1. What the table contains (subject matter)
            2. Key columns/dimensions
            3. Approximate size (rows/columns)
            4. Any notable patterns or categories
            
            Table:
            {table}
            
            Context (if available): {context}
            
            Summary (2-3 sentences):
            """
        )
        
        summary = self.llm.invoke(
            prompt.format(table=table_md, context=context)
        )
        
        return summary
    
    def process_tables(self, parsed_data: List[Dict]) -> List[Dict]:
        """Process all tables from parsed PDFs."""
        processed_tables = []
        
        for doc_data in parsed_data:
            pdf_path = doc_data["pdf_path"]
            
            for table_info in doc_data["tables"]:
                # Extract to DataFrame
                df = self.extract_table_to_dataframe(table_info["html"])
                
                if df.empty:
                    continue
                
                # Normalize
                df = self.normalize_table(df)
                
                # Generate summary
                summary = self.generate_table_summary(
                    df, 
                    context=f"From document: {Path(pdf_path).name}"
                )
                
                processed_tables.append({
                    "pdf_path": pdf_path,
                    "table_id": f"{Path(pdf_path).stem}_table_{len(processed_tables)}",
                    "page_number": table_info.get("page_number"),
                    "dataframe": df,
                    "summary": summary,
                })
        
        return processed_tables
```

---

## Phase 2: Dual-Index Creation

### Step 2.1: DOC Index (Vector Store)

Create `src/indexing/vector_store.py`:

```python
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_community import BigQueryVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import os

class VectorIndexBuilder:
    def __init__(self, project_id: str, dataset_name: str = "rag_dataset"):
        self.project_id = project_id
        self.dataset_name = dataset_name
        self.embeddings = VertexAIEmbeddings(model_name="text-embedding-005")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def create_documents_from_text(self, parsed_data: List[Dict]) -> List[Document]:
        """Create LangChain Documents from text elements."""
        documents = []
        
        for doc_data in parsed_data:
            text_elements = doc_data["text_elements"]
            pdf_path = doc_data["pdf_path"]
            
            for element in text_elements:
                # Chunk text
                chunks = self.text_splitter.split_text(str(element))
                
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": pdf_path,
                            "page_number": getattr(element.metadata, 'page_number', None),
                            "element_type": "text",
                            "chunk_id": i,
                        }
                    )
                    documents.append(doc)
        
        return documents
    
    def create_documents_from_table_summaries(self, processed_tables: List[Dict]) -> List[Document]:
        """Create Documents from table summaries."""
        documents = []
        
        for table_info in processed_tables:
            doc = Document(
                page_content=table_info["summary"],
                metadata={
                    "source": table_info["pdf_path"],
                    "page_number": table_info.get("page_number"),
                    "element_type": "table_summary",
                    "table_id": table_info["table_id"],
                }
            )
            documents.append(doc)
        
        return documents
    
    def build_index(self, text_docs: List[Document], table_summary_docs: List[Document]):
        """Build vector index in BigQuery."""
        all_docs = text_docs + table_summary_docs
        
        # Create BigQuery vector store
        vector_store = BigQueryVectorStore(
            project_id=self.project_id,
            dataset_name=self.dataset_name,
            table_name="doc_index",
            embedding=self.embeddings,
        )
        
        # Add documents
        print(f"Indexing {len(all_docs)} documents...")
        vector_store.add_documents(all_docs)
        
        print("Vector index created successfully!")
        return vector_store
```

### Step 2.2: FACTS Store (BigQuery)

Create `src/indexing/facts_store.py`:

```python
from google.cloud import bigquery
import pandas as pd
from typing import List, Dict
import os

class FactsStoreBuilder:
    def __init__(self, project_id: str, dataset_name: str = "rag_dataset"):
        self.client = bigquery.Client(project=project_id)
        self.dataset_name = dataset_name
        self._create_dataset()
    
    def _create_dataset(self):
        """Create BigQuery dataset if it doesn't exist."""
        dataset_id = f"{self.client.project}.{self.dataset_name}"
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"  # Or your preferred location
        
        try:
            self.client.create_dataset(dataset, exists_ok=True)
            print(f"Dataset {dataset_id} ready")
        except Exception as e:
            print(f"Dataset creation: {e}")
    
    def store_table(self, table_info: Dict, table_name: str = None):
        """Store a table in BigQuery."""
        df = table_info["dataframe"]
        table_id = table_info["table_id"]
        
        if table_name is None:
            # Sanitize table name
            table_name = f"table_{table_id.replace('-', '_').replace('.', '_')}"
        
        full_table_id = f"{self.client.project}.{self.dataset_name}.{table_name}"
        
        # Add metadata columns
        df_with_metadata = df.copy()
        df_with_metadata['_source_pdf'] = table_info["pdf_path"]
        df_with_metadata['_table_id'] = table_id
        df_with_metadata['_page_number'] = table_info.get("page_number", None)
        
        # Upload to BigQuery
        job = self.client.load_table_from_dataframe(
            df_with_metadata, 
            full_table_id
        )
        job.result()  # Wait for completion
        
        print(f"Stored table {table_id} in {full_table_id}")
        return full_table_id
    
    def store_all_tables(self, processed_tables: List[Dict]):
        """Store all tables in BigQuery."""
        stored_tables = []
        
        for table_info in processed_tables:
            try:
                table_id = self.store_table(table_info)
                stored_tables.append({
                    "table_id": table_info["table_id"],
                    "bigquery_table": table_id,
                    "pdf_path": table_info["pdf_path"],
                })
            except Exception as e:
                print(f"Error storing table {table_info['table_id']}: {e}")
                continue
        
        return stored_tables
```

---

## Phase 3: RAG Agent Implementation

### Step 3.1: Agentic Router

Create `src/retrieval/rag_agent.py`:

```python
from langchain_google_vertexai import VertexAI
from langchain_google_community import BigQueryVectorStore
from langchain.agents import AgentExecutor, create_sql_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from google.cloud import bigquery
from typing import List, Dict
import flashrank

class RAGAgent:
    def __init__(self, project_id: str, dataset_name: str, vector_store: BigQueryVectorStore):
        self.llm = VertexAI(model_name="gemini-1.5-pro")
        self.project_id = project_id
        self.dataset_name = dataset_name
        self.vector_store = vector_store
        self.bq_client = bigquery.Client(project=project_id)
        self.reranker = flashrank.RerankRequest()
        
    def classify_query(self, query: str) -> str:
        """Classify query as semantic, quantitative, or hybrid."""
        prompt = f"""
        Classify this query into one of these categories:
        1. "semantic" - Questions about concepts, explanations, descriptions
        2. "quantitative" - Questions requiring numbers, calculations, filtered data from tables
        3. "hybrid" - Questions that need both semantic understanding and precise data
        
        Query: {query}
        
        Category (respond with only the category name):
        """
        
        category = self.llm.invoke(prompt).strip().lower()
        return category
    
    def semantic_retrieval(self, query: str, k: int = 10) -> List[Dict]:
        """Retrieve from vector store."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(query)
        
        # Re-rank
        reranked = self._rerank(query, docs)
        
        return reranked
    
    def _rerank(self, query: str, docs: List) -> List[Dict]:
        """Re-rank retrieved documents."""
        # Using FlashRank
        from flashrank import Ranker
        
        ranker = Ranker()
        results = ranker.rank(query, [doc.page_content for doc in docs])
        
        # Map back to documents
        reranked_docs = []
        for result in results:
            idx = result['original_index']
            reranked_docs.append({
                "content": docs[idx].page_content,
                "metadata": docs[idx].metadata,
                "score": result['score']
            })
        
        return reranked_docs
    
    def sql_retrieval(self, query: str) -> str:
        """Generate and execute SQL query."""
        # Get list of available tables
        tables = self._list_tables()
        
        prompt = f"""
        Given this question: {query}
        
        Available tables in BigQuery dataset {self.dataset_name}:
        {', '.join(tables)}
        
        Generate a SQL query to answer this question. 
        Use standard SQL syntax for BigQuery.
        Only query tables that are relevant.
        
        SQL Query:
        """
        
        sql_query = self.llm.invoke(prompt).strip()
        
        # Remove markdown code blocks if present
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        # Execute query
        try:
            query_job = self.bq_client.query(sql_query)
            results = query_job.result()
            
            # Format results
            rows = [dict(row) for row in results]
            return self._format_sql_results(rows)
        except Exception as e:
            return f"Error executing SQL: {e}. Generated SQL: {sql_query}"
    
    def _list_tables(self) -> List[str]:
        """List all tables in the dataset."""
        dataset_ref = self.bq_client.dataset(self.dataset_name)
        tables = list(self.bq_client.list_tables(dataset_ref))
        return [table.table_id for table in tables]
    
    def _format_sql_results(self, rows: List[Dict]) -> str:
        """Format SQL results as natural language."""
        if not rows:
            return "No results found."
        
        # Convert to DataFrame for better formatting
        import pandas as pd
        df = pd.DataFrame(rows)
        
        return f"Query results:\n{df.to_string()}"
    
    def answer(self, query: str) -> Dict:
        """Main method to answer a query."""
        # Classify query
        category = self.classify_query(query)
        
        context_parts = []
        sources = []
        
        if category in ["semantic", "hybrid"]:
            # Semantic retrieval
            semantic_results = self.semantic_retrieval(query, k=10)
            context_parts.append("Text Context:\n" + "\n\n".join([
                f"[Source: {r['metadata'].get('source', 'unknown')}, Page {r['metadata'].get('page_number', '?')}]\n{r['content']}"
                for r in semantic_results[:5]  # Top 5 after reranking
            ]))
            sources.extend([r['metadata'] for r in semantic_results[:5]])
        
        if category in ["quantitative", "hybrid"]:
            # SQL retrieval
            sql_result = self.sql_retrieval(query)
            context_parts.append(f"Table Data:\n{sql_result}")
            sources.append({"type": "sql_query", "result": sql_result})
        
        # Generate final answer
        context = "\n\n".join(context_parts)
        
        answer_prompt = f"""
        Answer the following question using ONLY the provided context.
        Cite your sources explicitly (document name, page number, table ID).
        If the question requires calculations, show your work step-by-step.
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:
        """
        
        answer = self.llm.invoke(answer_prompt)
        
        return {
            "answer": answer,
            "sources": sources,
            "query_category": category,
        }
```

---

## Phase 4: Evaluation Framework

### Step 4.1: Evaluation Script

Create `src/evaluation/evaluate.py`:

```python
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
from src.retrieval.rag_agent import RAGAgent
from typing import List, Dict

class RAGEvaluator:
    def __init__(self, rag_agent: RAGAgent):
        self.rag_agent = rag_agent
    
    def load_golden_dataset(self, csv_path: str) -> pd.DataFrame:
        """Load golden dataset from CSV."""
        df = pd.read_csv(csv_path)
        return df
    
    def run_evaluation(self, golden_df: pd.DataFrame) -> Dict:
        """Run RAG pipeline on all questions and evaluate."""
        results = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }
        
        for _, row in golden_df.iterrows():
            query = row["question"]
            expected = row["expected_output"]
            
            print(f"Processing: {query[:50]}...")
            
            # Get answer from RAG agent
            response = self.rag_agent.answer(query)
            
            # Extract contexts
            contexts = []
            for source in response["sources"]:
                if "content" in source:
                    contexts.append(source["content"])
                elif "result" in source:
                    contexts.append(source["result"])
            
            results["question"].append(query)
            results["answer"].append(response["answer"])
            results["contexts"].append(contexts)
            results["ground_truth"].append(expected)
        
        # Create dataset for RAGAS
        dataset = Dataset.from_dict(results)
        
        # Evaluate
        scores = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ]
        )
        
        return {
            "scores": scores,
            "results": results,
        }
    
    def generate_report(self, evaluation_results: Dict, output_path: str = "evaluation_report.html"):
        """Generate HTML evaluation report."""
        scores = evaluation_results["scores"]
        
        # Create report
        report = f"""
        <html>
        <head><title>RAG Evaluation Report</title></head>
        <body>
        <h1>RAG Evaluation Report</h1>
        
        <h2>Overall Metrics</h2>
        <ul>
        <li>Faithfulness: {scores['faithfulness']:.3f}</li>
        <li>Answer Relevancy: {scores['answer_relevancy']:.3f}</li>
        <li>Context Precision: {scores['context_precision']:.3f}</li>
        <li>Context Recall: {scores['context_recall']:.3f}</li>
        </ul>
        
        <h2>Per-Question Results</h2>
        <table border="1">
        <tr>
            <th>Question</th>
            <th>Generated Answer</th>
            <th>Expected Answer</th>
        </tr>
        """
        
        for i, (q, a, gt) in enumerate(zip(
            evaluation_results["results"]["question"],
            evaluation_results["results"]["answer"],
            evaluation_results["results"]["ground_truth"]
        )):
            report += f"""
            <tr>
                <td>{q[:100]}...</td>
                <td>{a[:200]}...</td>
                <td>{str(gt)[:200]}...</td>
            </tr>
            """
        
        report += """
        </table>
        </body>
        </html>
        """
        
        with open(output_path, "w") as f:
            f.write(report)
        
        print(f"Report saved to {output_path}")

if __name__ == "__main__":
    # Load configuration
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    
    # Initialize components (assuming they're already built)
    from src.indexing.vector_store import VectorIndexBuilder
    from src.retrieval.rag_agent import RAGAgent
    
    # Load vector store (assume it exists)
    vector_store = BigQueryVectorStore(
        project_id=project_id,
        dataset_name="rag_dataset",
        table_name="doc_index",
        embedding=VertexAIEmbeddings(model_name="text-embedding-005"),
    )
    
    # Create RAG agent
    agent = RAGAgent(project_id, "rag_dataset", vector_store)
    
    # Run evaluation
    evaluator = RAGEvaluator(agent)
    golden_df = evaluator.load_golden_dataset("artifacts/questions.csv")
    
    results = evaluator.run_evaluation(golden_df)
    evaluator.generate_report(results)
```

---

## Main Pipeline Script

Create `main.py`:

```python
import os
from dotenv import load_dotenv
from src.parsing.pdf_parser import PDFParser
from src.parsing.table_processor import TableProcessor
from src.indexing.vector_store import VectorIndexBuilder
from src.indexing.facts_store import FactsStoreBuilder
from src.retrieval.rag_agent import RAGAgent

load_dotenv()

def main():
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    dataset_name = "rag_dataset"
    
    # Phase 1: Parse PDFs
    print("Phase 1: Parsing PDFs...")
    parser = PDFParser()
    parsed_data = parser.process_all_pdfs("artifacts/1")
    parsed_data.extend(parser.process_all_pdfs("artifacts/2"))
    
    # Phase 1.2: Process tables
    print("Phase 1.2: Processing tables...")
    table_processor = TableProcessor()
    processed_tables = table_processor.process_tables(parsed_data)
    
    # Phase 2: Build indexes
    print("Phase 2: Building indexes...")
    
    # DOC Index
    vector_builder = VectorIndexBuilder(project_id, dataset_name)
    text_docs = vector_builder.create_documents_from_text(parsed_data)
    table_summary_docs = vector_builder.create_documents_from_table_summaries(processed_tables)
    vector_store = vector_builder.build_index(text_docs, table_summary_docs)
    
    # FACTS Store
    facts_builder = FactsStoreBuilder(project_id, dataset_name)
    facts_builder.store_all_tables(processed_tables)
    
    print("Indexing complete!")
    
    # Phase 3: Test RAG agent
    print("Phase 3: Testing RAG agent...")
    agent = RAGAgent(project_id, dataset_name, vector_store)
    
    # Test query
    test_query = "List all rating plan rules"
    response = agent.answer(test_query)
    print(f"Query: {test_query}")
    print(f"Answer: {response['answer']}")
    print(f"Sources: {len(response['sources'])} sources found")
    
    # Phase 4: Evaluation (optional, run separately)
    print("\nTo run evaluation, use: python src/evaluation/evaluate.py")

if __name__ == "__main__":
    main()
```

---

## Running the Pipeline

```bash
# 1. Set up environment
source venv/bin/activate

# 2. Run main pipeline
python main.py

# 3. Run evaluation
python src/evaluation/evaluate.py
```

---

## Troubleshooting

### Common Issues

1. **BigQuery Permissions**: Ensure service account has BigQuery Admin role
2. **Vertex AI Quotas**: Check API quotas in GCP Console
3. **Table Parsing Errors**: Some tables may need manual preprocessing
4. **Memory Issues**: Process PDFs in batches for large files

### Optimization Tips

1. **Chunk Size**: Adjust based on average document length
2. **Re-ranking**: Start with FlashRank, upgrade to Cohere if needed
3. **SQL Generation**: Fine-tune prompts for better SQL accuracy
4. **Caching**: Cache embeddings and summaries to avoid re-computation

---

## Next Steps

1. Test on 2-3 sample PDFs first
2. Validate table extraction quality
3. Tune retrieval parameters (k, reranking)
4. Run full evaluation on golden dataset
5. Compare vs. baseline (classic RAG)

