"""
RAG Query Testing Script

This script tests the RAG system using pre-built indices.
It does NOT rebuild indices - assumes they already exist from build_indices.py.

Usage:
  python3 query_rag.py                    # Interactive mode
  python3 query_rag.py "your question"    # Single query
  python3 query_rag.py --file questions.csv  # Batch queries from CSV
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.retrieval.rag_agent import RAGAgent


def interactive_mode(agent: RAGAgent) -> None:
    """Interactive query mode."""
    print("\n" + "="*60)
    print("INTERACTIVE RAG QUERY MODE")
    print("="*60)
    print("Enter your questions (type 'quit' or 'exit' to stop):\n")
    
    while True:
        try:
            question = input("Question: ").strip()
            if not question:
                continue
            if question.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            
            print("\n" + "-"*60)
            resp = agent.answer(question)
            print(f"\nQuery Category: {resp['query_category']}")
            print(f"\nAnswer:\n{resp['answer']}")
            print("-"*60 + "\n")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}\n")


def single_query(agent: RAGAgent, question: str) -> None:
    """Run a single query."""
    print("\n" + "="*60)
    print("RAG QUERY")
    print("="*60)
    print(f"Question: {question}\n")
    
    resp = agent.answer(question)
    
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Query Category: {resp['query_category']}")
    print(f"\nAnswer:\n{resp['answer']}")
    print("="*60)


def batch_queries(agent: RAGAgent, csv_path: Path) -> None:
    """Run queries from a CSV file."""
    import pandas as pd
    
    print("\n" + "="*60)
    print("BATCH QUERY PROCESSING")
    print("="*60)
    print(f"Reading questions from: {csv_path}\n")
    
    if not csv_path.exists():
        print(f"✗ Error: File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    if 'question' not in df.columns and 'query' not in df.columns:
        print("✗ Error: CSV must have 'question' or 'query' column")
        return
    
    question_col = 'question' if 'question' in df.columns else 'query'
    questions = df[question_col].dropna().tolist()
    
    print(f"Found {len(questions)} questions to process\n")
    
    results = []
    for idx, question in enumerate(questions, 1):
        print(f"[{idx}/{len(questions)}] Processing: {question[:60]}...")
        try:
            resp = agent.answer(question)
            results.append({
                'question': question,
                'category': resp['query_category'],
                'answer': resp['answer'],
            })
            print(f"  ✓ Category: {resp['query_category']}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'question': question,
                'category': 'error',
                'answer': f"Error: {e}",
            })
    
    # Save results
    output_path = csv_path.parent / f"{csv_path.stem}_results.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")


def main() -> None:
    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise RuntimeError("Missing GOOGLE_CLOUD_PROJECT environment variable.")
    
    # Verify GCP authentication
    try:
        from src.utils.auth_helper import verify_gcp_auth, print_auth_setup_instructions
        is_valid, error_msg = verify_gcp_auth(project_id)
        if not is_valid:
            print(f"\n❌ GCP Authentication Error:\n{error_msg}\n")
            print_auth_setup_instructions()
            raise RuntimeError(f"GCP authentication failed: {error_msg}")
        # Success message is printed by verify_gcp_auth if default key is used
    except ImportError:
        # auth_helper not available, skip verification
        pass

    dataset_name = "rag_dataset"

    # Initialize RAG Agent (loads existing indices)
    print("Initializing RAGAgent (loading existing indices)...")
    agent = RAGAgent(project_id=project_id, dataset_name=dataset_name)
    print("✓ RAGAgent initialized successfully\n")

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test RAG system with pre-built indices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 query_rag.py
  python3 query_rag.py "List all rating plan rules"
  python3 query_rag.py --file artifacts/questions.csv
        """
    )
    parser.add_argument(
        'question',
        nargs='?',
        help='Single question to ask (if not provided, enters interactive mode)'
    )
    parser.add_argument(
        '--file', '-f',
        type=Path,
        help='CSV file with questions (must have "question" or "query" column)'
    )
    
    args = parser.parse_args()

    # Determine mode
    if args.file:
        batch_queries(agent, args.file)
    elif args.question:
        single_query(agent, args.question)
    else:
        interactive_mode(agent)


if __name__ == "__main__":
    main()

