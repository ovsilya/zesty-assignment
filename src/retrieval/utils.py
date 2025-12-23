"""
Utility functions for RAG agent operations.

These are general-purpose functions that don't require instance state
and can be easily tested and reused.
"""

import re
from typing import List, Dict, Any


def table_id_to_bigquery_table_name(table_id: str) -> str:
    """Convert a table_id from metadata to a BigQuery-compatible table name.
    
    Args:
        table_id: Table ID from metadata (e.g., "(213128717-179157013)-MCY Rate Filing Data Summary - SFFC_table_1")
        
    Returns:
        BigQuery table name (e.g., "table__213128717_179157013__MCY_Rate_Filing_Data_Summary___SFFC_table_1")
    """
    # Remove parentheses and replace special characters
    # Format: (numbers-numbers)-Name_table_N -> table__numbers_numbers__Name_table_N
    cleaned = table_id.replace("(", "").replace(")", "")
    
    # Replace hyphens with underscores (but preserve structure)
    # Split by pattern: numbers-numbers or numbers_numbers
    parts = re.split(r'[-_]', cleaned, maxsplit=2)
    
    if len(parts) >= 3:
        # Format: numbers, numbers, rest
        prefix = f"table__{parts[0]}_{parts[1]}__"
        rest = "_".join(parts[2:])
        # Replace remaining special chars
        rest = re.sub(r'[^a-zA-Z0-9_]', '_', rest)
        return prefix + rest
    
    # Fallback: just sanitize the whole thing
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', cleaned)
    return f"table__{sanitized}"


def extract_table_ids_from_results(semantic_results: List[Dict[str, Any]]) -> List[str]:
    """Extract table_ids from semantic retrieval results that are table summaries.
    
    Args:
        semantic_results: List of semantic search results with metadata
        
    Returns:
        List of table IDs found in the results
    """
    table_ids = []
    for result in semantic_results:
        meta = result.get("metadata", {})
        if meta.get("element_type") == "table_summary":
            table_id = meta.get("table_id")
            if table_id:
                table_ids.append(table_id)
    return table_ids


def extract_specific_values_from_query(query: str) -> List[str]:
    """Extract specific values from a query that can be used for table search.
    
    Extracts:
    - ZIP codes (5-digit numbers)
    - Territory numbers (3-digit numbers after "territory")
    - Percentages (numbers with %)
    - Model names (capitalized phrases that might be motorcycle models)
    
    Args:
        query: The user's question
        
    Returns:
        List of extracted values
    """
    values = []
    
    # Extract ZIP codes (5-digit numbers, often mentioned with "ZIP" or "zip code")
    zip_pattern = r'\b(?:zip|zip code|zips)\s*(?:code\s*)?(\d{5})\b'
    zip_matches = re.findall(zip_pattern, query, re.IGNORECASE)
    values.extend(zip_matches)
    
    # Extract territory numbers (3-digit numbers, often after "territory" or "terr")
    territory_pattern = r'\b(?:territory|terr)\s*(\d{1,3})\b'
    territory_matches = re.findall(territory_pattern, query, re.IGNORECASE)
    values.extend(territory_matches)
    
    # Extract percentages (numbers with % sign)
    percent_pattern = r'([+-]?\d+\.?\d*)\s*%'
    percent_matches = re.findall(percent_pattern, query)
    values.extend(percent_matches)
    
    # Extract potential model names (capitalized multi-word phrases)
    # Look for patterns like "Ducati Panigale V4 R" or "Honda Grom ABS"
    model_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
    model_matches = re.findall(model_pattern, query)
    # Filter out common words that aren't model names
    common_words = {'State', 'Farm', 'Fire', 'Casualty', 'Company', 'Policyholder', 
                    'Resident', 'Living', 'West', 'East', 'North', 'South', 'Boulevard',
                    'Street', 'Avenue', 'Road', 'Drive', 'Coverage', 'Comprehensive',
                    'Collision', 'Rating', 'Group', 'Motorcycle', 'Model', 'Year'}
    for match in model_matches:
        words = match.split()
        # If it's 2+ words and not all common words, likely a model name
        if len(words) >= 2 and not all(w in common_words for w in words):
            values.append(match)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_values = []
    for v in values:
        if v.lower() not in seen:
            seen.add(v.lower())
            unique_values.append(v)
    
    return unique_values


def extract_geographic_terms(query: str) -> List[str]:
    """Extract geographic terms (street names, directions) from a query.
    
    Args:
        query: The user's question
        
    Returns:
        List of extracted geographic terms
    """
    geographic_terms = []
    geographic_patterns = [
        r'(?:west|east|north|south)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'(?:boulevard|street|avenue|road|drive|way)\s+([A-Z][a-z]+)',
    ]
    for pattern in geographic_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        geographic_terms.extend(matches)
    return geographic_terms


def are_likely_splits(table1: str, table2: str, schema1: Dict, schema2: Dict) -> bool:
    """Check if two tables are likely splits of the same logical table.
    
    Args:
        table1: First table ID
        table2: Second table ID
        schema1: Schema dict for first table
        schema2: Schema dict for second table
        
    Returns:
        True if tables are likely splits
    """
    # Check if from same PDF (extract from table name or metadata)
    def extract_pdf_id(table_name: str) -> str:
        # Format: table__{numbers}_{numbers}__{pdf_name}...
        parts = table_name.split('__')
        if len(parts) >= 3:
            return f"{parts[1]}_{parts[2]}"  # First two number groups
        return table_name.split('_')[0] if '_' in table_name else table_name
    
    pdf_id1 = extract_pdf_id(table1)
    pdf_id2 = extract_pdf_id(table2)
    
    # If same PDF identifier, likely splits
    if pdf_id1 == pdf_id2 and len(pdf_id1) > 5:  # Meaningful identifier
        return True
    
    # Also check if table IDs are similar (consecutive table numbers)
    # Format: ..._table_{N}_page_{M}
    match1 = re.search(r'table_(\d+)', table1)
    match2 = re.search(r'table_(\d+)', table2)
    if match1 and match2:
        num1 = int(match1.group(1))
        num2 = int(match2.group(1))
        # If consecutive table numbers from same PDF, likely splits
        if abs(num1 - num2) <= 2 and pdf_id1 == pdf_id2:
            return True
    
    return False


def sanitize_table_name(name: str) -> str:
    """Sanitize a table name for BigQuery compatibility.
    
    Args:
        name: Original table name
        
    Returns:
        Sanitized table name safe for BigQuery
    """
    # Replace special characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized

