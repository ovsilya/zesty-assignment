#!/usr/bin/env python3
"""
Compare test results with expected answers from questions.csv
"""

import pandas as pd
from pathlib import Path
import re

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if pd.isna(text):
        return ""
    return " ".join(str(text).lower().split())

def extract_rules_from_text(text: str) -> set:
    """Extract rule names from text (for EF_1)."""
    rules = set()
    
    # Pattern 1: "Rule C-X: Name" or "Rule C-X Name"
    pattern1 = r'rule\s+c-?\d+[:\s]+([^,\n*]+?)(?:,|\n|$)'
    matches = re.findall(pattern1, text, re.IGNORECASE)
    rules.update([m.strip().lower() for m in matches])
    
    # Pattern 2: "* Rule Name" (bullet points)
    pattern2 = r'\*\s+([^,\n*]+?)(?:,|\n|$)'
    matches = re.findall(pattern2, text, re.IGNORECASE)
    rules.update([m.strip().lower() for m in matches])
    
    # Pattern 3: Extract key phrases that look like rule names
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('*') or 'rule' in line.lower():
            cleaned = re.sub(r'^\*\s*', '', line)
            cleaned = re.sub(r'^rule\s+c-?\d+[:\s]+', '', cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.split(',')[0].split('(')[0].strip()
            if len(cleaned) > 5:
                rules.add(cleaned.lower())
    
    return rules

def check_ef1(predicted: str, expected: str) -> dict:
    """Check EF_1 (List all rating plan rules)."""
    pred_norm = normalize_text(predicted)
    exp_norm = normalize_text(expected)
    
    # Extract rules from both
    pred_rules = extract_rules_from_text(predicted)
    exp_rules = extract_rules_from_text(expected)
    
    # Also check for key rule names directly
    key_rules = [
        "limits of liability", "rating perils", "base rates", "policy type factor",
        "policy tier", "amount of insurance", "deductibles", "hurricane deductibles",
        "windstorm", "hail deductibles", "territory determination", "distance to coast",
        "public protection class", "age of home", "year built", "account discount",
        "roof type", "dwelling usage", "increased limits", "protective device",
        "affinity discount", "association discount", "oil tank", "pool factor",
        "trampoline", "roof condition", "tree overhang", "solar panel",
        "secondary heating", "windstorm mitigation", "endorsement combination",
        "loss history", "claims free", "underwriting experience", "minimum premium"
    ]
    
    found_rules = []
    missing_rules = []
    for rule in key_rules:
        if rule in pred_norm:
            found_rules.append(rule)
        else:
            missing_rules.append(rule)
    
    # Count rules found
    rule_count = len([r for r in key_rules if r in pred_norm])
    total_rules = len(key_rules)
    
    return {
        "question": "EF_1",
        "found_rules": found_rules,
        "missing_rules": missing_rules,
        "rule_count": rule_count,
        "total_rules": total_rules,
        "coverage": rule_count / total_rules if total_rules > 0 else 0,
        "status": "CORRECT" if rule_count >= total_rules * 0.9 else "PARTIAL" if rule_count >= total_rules * 0.7 else "INCORRECT"
    }

def check_ef2(predicted: str, expected: str) -> dict:
    """Check EF_2 (Territory and GRG comparison)."""
    pred_norm = normalize_text(predicted)
    exp_norm = normalize_text(expected)
    
    checks = {
        "says_yes": "yes" in pred_norm and "no" not in pred_norm[:200],
        "territory_118": "territory 118" in pred_norm or ("118" in pred_norm and "territory" in pred_norm),
        "territory_117": "territory 117" in pred_norm or ("117" in pred_norm and "territory" in pred_norm),
        "rate_0_305": "0.305" in pred_norm or "0.305%" in pred_norm,
        "rate_neg_0_133": "-0.133" in pred_norm or "-0.133%" in pred_norm,
        "west_nellis": "west of nellis" in pred_norm or ("west" in pred_norm and "nellis" in pred_norm),
        "east_nellis": "east of nellis" in pred_norm or ("east" in pred_norm and "nellis" in pred_norm),
        "grg_difference_36": "36" in pred_norm or ("exceed" in pred_norm and "30" in pred_norm and any(x in pred_norm for x in ["36", "51", "15"])),
        "grg_51": "51" in pred_norm or "051" in pred_norm,
        "grg_15": "15" in pred_norm or "015" in pred_norm,
    }
    
    passed = sum(checks.values())
    total = len(checks)
    score = passed / total
    
    return {
        "question": "EF_2",
        "checks": checks,
        "passed": passed,
        "total": total,
        "score": score,
        "status": "CORRECT" if score >= 0.9 else "PARTIAL" if score >= 0.7 else "INCORRECT"
    }

def check_ef3(predicted: str, expected: str) -> dict:
    """Check EF_3 (Hurricane premium calculation)."""
    pred_norm = normalize_text(predicted)
    exp_norm = normalize_text(expected)
    
    # Expected answer is "$604"
    expected_value = "604"
    
    # Check if answer contains the expected value
    has_604 = "604" in pred_norm or "$604" in predicted
    
    # Check if answer says it cannot calculate
    cannot_calculate = any(phrase in pred_norm for phrase in [
        "cannot calculate", "cannot compute", "not possible", "missing",
        "not available", "not provided", "unable to", "do not have enough"
    ])
    
    # Check if it mentions finding the values but not calculating
    mentions_values = any(phrase in pred_norm for phrase in [
        "base rate", "deductible factor", "mandatory hurricane", "2%"
    ])
    
    return {
        "question": "EF_3",
        "has_604": has_604,
        "cannot_calculate": cannot_calculate,
        "mentions_values": mentions_values,
        "expected": "$604",
        "status": "CORRECT" if has_604 and not cannot_calculate else "INCORRECT"
    }

def main():
    project_root = Path(__file__).parent
    questions_file = project_root / "artifacts" / "questions.csv"
    results_file = project_root / "artifacts" / "questions_results.csv"
    
    # Read files
    questions_df = pd.read_csv(questions_file)
    results_df = pd.read_csv(results_file)
    
    print("="*80)
    print("TEST RESULTS COMPARISON")
    print("="*80)
    print()
    
    # Create mapping from question to expected answer
    question_map = {}
    for _, row in questions_df.iterrows():
        qid = row.get('id', '')
        question = row.get('question', '')
        expected = row.get('expected_output', '')
        question_map[qid] = {
            'question': question,
            'expected': expected
        }
    
    # Process each result
    all_results = []
    
    for _, row in results_df.iterrows():
        question_text = row.get('question', '')
        predicted = row.get('answer', '')
        
        # Find matching question ID
        qid = None
        for q_id, q_data in question_map.items():
            if q_data['question'].lower() in question_text.lower() or question_text.lower() in q_data['question'].lower():
                qid = q_id
                break
        
        if not qid:
            print(f"⚠ Warning: Could not match question: {question_text[:50]}...")
            continue
        
        expected = question_map[qid]['expected']
        
        print(f"\n{'='*80}")
        print(f"Question ID: {qid}")
        print(f"{'='*80}")
        print(f"Question: {question_text[:100]}...")
        print()
        
        # Evaluate based on question type
        if qid == "EF_1":
            result = check_ef1(predicted, expected)
            print(f"Status: {result['status']}")
            print(f"Rules Found: {result['rule_count']}/{result['total_rules']} ({result['coverage']:.1%})")
            if result['missing_rules']:
                print(f"Missing Rules ({len(result['missing_rules'])}): {', '.join(result['missing_rules'][:10])}")
        
        elif qid == "EF_2":
            result = check_ef2(predicted, expected)
            print(f"Status: {result['status']}")
            print(f"Checks Passed: {result['passed']}/{result['total']} ({result['score']:.1%})")
            print("\nDetailed Checks:")
            for check_name, passed in result['checks'].items():
                status = "✓" if passed else "✗"
                print(f"  {status} {check_name}")
        
        elif qid == "EF_3":
            result = check_ef3(predicted, expected)
            print(f"Status: {result['status']}")
            print(f"Expected: {result['expected']}")
            print(f"Has $604: {result['has_604']}")
            print(f"Says cannot calculate: {result['cannot_calculate']}")
            print(f"Mentions finding values: {result['mentions_values']}")
            if result['cannot_calculate'] and result['mentions_values']:
                print("\n⚠ Note: LLM found the values but didn't perform the calculation")
        
        all_results.append({
            'question_id': qid,
            'status': result['status'],
            **result
        })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    correct = sum(1 for r in all_results if r['status'] == 'CORRECT')
    partial = sum(1 for r in all_results if r['status'] == 'PARTIAL')
    incorrect = sum(1 for r in all_results if r['status'] == 'INCORRECT')
    total = len(all_results)
    
    print(f"Total Questions: {total}")
    print(f"Correct: {correct} ({correct/total:.1%})")
    print(f"Partial: {partial} ({partial/total:.1%})")
    print(f"Incorrect: {incorrect} ({incorrect/total:.1%})")
    print()
    
    for r in all_results:
        print(f"  {r['question_id']}: {r['status']}")
    
    # Detailed breakdown
    print("\n" + "="*80)
    print("DETAILED BREAKDOWN")
    print("="*80)
    
    for r in all_results:
        qid = r['question_id']
        print(f"\n{qid}:")
        if qid == "EF_1":
            print(f"  Coverage: {r['rule_count']}/{r['total_rules']} rules found ({r['coverage']:.1%})")
        elif qid == "EF_2":
            print(f"  Score: {r['passed']}/{r['total']} checks passed ({r['score']:.1%})")
        elif qid == "EF_3":
            print(f"  Has answer: {r['has_604']}")
            print(f"  Cannot calculate: {r['cannot_calculate']}")

if __name__ == "__main__":
    main()

