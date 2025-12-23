# Evaluation Metrics Documentation

This document explains the evaluation metrics used to assess the performance of the RAG (Retrieval-Augmented Generation) system.

## Overview

The evaluation framework uses a combination of **custom metrics** and **RAGAS metrics** (when available) to comprehensively assess the quality of generated answers against expected outputs. The metrics are designed to evaluate different aspects of answer quality, from exact correctness to completeness and source attribution.

---

## Custom Metrics

### 1. Exact Match

**Purpose**: Determines if the generated answer exactly matches the expected output.

**Calculation**:
- Normalizes both answers (lowercase, whitespace normalization)
- Compares character-by-character after normalization
- Returns `True` if identical, `False` otherwise

**Interpretation**:
- **True**: The answer is exactly correct (allowing for case/whitespace differences)
- **False**: The answer differs from expected in any way

**Use Case**: Best for questions with precise, unambiguous answers (e.g., calculations, specific values).

**Limitations**: Very strict - may penalize correct answers that are phrased differently or include additional context.

---

### 2. Keyword Coverage

**Purpose**: Measures how many important keywords from the expected answer appear in the generated answer.

**Calculation**:
1. Extracts keywords from both answers:
   - Capitalized phrases (e.g., "Territory 118", "Rule C-1")
   - Numbers and percentages (e.g., "293", "0.305%", "$604")
   - Important words (4+ characters, excluding common stop words)
2. Calculates: `(matched keywords) / (total expected keywords)`
3. Returns a score between 0.0 and 1.0

**Interpretation**:
- **1.0 (100%)**: All important keywords from expected answer are present
- **0.5 (50%)**: Half of the important keywords are present
- **0.0 (0%)**: None of the important keywords are present

**Use Case**: Useful for list questions, descriptive answers, and answers where specific terms matter more than exact phrasing.

**Example**:
- Expected: "Territory 118 has a rate change of 0.305%"
- Generated: "The rate change for Territory 118 is 0.305%"
- Coverage: High (both contain "Territory 118" and "0.305%")

---

### 3. Number Match

**Purpose**: Specifically evaluates numeric accuracy in answers.

**Calculation**:
1. Extracts all numeric values from both answers (including dollar amounts, percentages)
2. Compares the sets of numbers
3. Calculates: `(matched numbers) / (total expected numbers)`
4. Returns a score between 0.0 and 1.0

**Interpretation**:
- **1.0 (100%)**: All numeric values match
- **0.5 (50%)**: Half of the numeric values match
- **0.0 (0%)**: No numeric values match

**Use Case**: Critical for calculation questions, rate changes, percentages, and any answer where numeric precision is essential.

**Example**:
- Expected: "$604"
- Generated: "The premium is $604"
- Match: 1.0 (the number 604 is present)

---

### 4. Answer Completeness

**Purpose**: Estimates how complete the generated answer is compared to the expected answer.

**Calculation**:
1. Compares word count: `min(generated_words / expected_words, 1.0)`
2. Calculates keyword coverage (see above)
3. Averages the two scores: `(length_score + keyword_score) / 2`

**Interpretation**:
- **1.0 (100%)**: Answer is as complete as expected (or more complete)
- **0.5 (50%)**: Answer is roughly half as complete
- **0.0 (0%)**: Answer is very incomplete

**Use Case**: Useful for detecting when answers are too brief or missing important information, especially for list questions or comprehensive explanations.

**Limitations**: A longer answer isn't always better - this metric should be combined with accuracy metrics.

---

### 5. Source Citation Quality

**Purpose**: Evaluates whether the answer includes proper source citations (tables, pages, documents).

**Calculation**:
1. Searches for source indicators in the generated answer:
   - "source:", "table:", "page", "document", "pdf"
   - "according to", "based on", "from"
2. Counts how many indicators are found
3. Scores: `min(found_indicators / 3.0, 1.0)`

**Interpretation**:
- **1.0 (100%)**: Answer includes 3+ source citations
- **0.67 (67%)**: Answer includes 2 source citations
- **0.33 (33%)**: Answer includes 1 source citation
- **0.0 (0%)**: Answer includes no source citations

**Use Case**: Important for traceability and verification - answers should cite where information came from.

---

## Question-Specific Metrics

The framework includes specialized metrics for specific question types:

### EF_1: List Question Metrics

**Purpose**: Evaluates completeness of list-type answers (e.g., "List all rating plan rules").

**Metrics**:
- **`rules_found`**: Number of rules correctly identified
- **`total_rules`**: Total number of rules in expected answer
- **`rules_coverage`**: `rules_found / total_rules`

**Calculation**:
- Extracts rule names from both answers (markdown list format: `* Rule Name`)
- Also checks for "Rule C-X" patterns
- Compares sets to find matches

**Interpretation**:
- **1.0 (100%)**: All rules found
- **0.8 (80%)**: 80% of rules found
- **0.0 (0%)**: No rules found

---

### EF_2: Territory/GRG Question Metrics

**Purpose**: Checks for specific facts in territory and GRG (Group Rating Group) questions.

**Fact Checks**:
- **`has_yes`**: Answer contains "yes" (for yes/no questions)
- **`has_territory_118`**: Mentions Territory 118
- **`has_territory_117`**: Mentions Territory 117
- **`has_rate_0_305`**: Contains rate 0.305%
- **`has_rate_neg_0_133`**: Contains rate -0.133%
- **`has_grg_51`**: Mentions GRG 51
- **`has_grg_15`**: Mentions GRG 15
- **`has_difference_36`**: Mentions difference of 36 (or exceeds 30)

**Overall Score**:
- **`fact_score`**: Average of all fact checks (0.0 to 1.0)

**Interpretation**:
- **1.0 (100%)**: All facts present
- **0.5 (50%)**: Half of the facts present
- **0.0 (0%)**: No facts present

---

### EF_3: Calculation Question Metrics

**Purpose**: Validates calculation questions (e.g., premium calculations).

**Metrics**:
- **`has_604`**: Contains the correct answer ($604)
- **`has_293`**: Contains the base rate (293)
- **`has_2_061`**: Contains the factor (2.061)
- **`has_calculation`**: Shows calculation steps (×, *, "multiply")
- **`cannot_calculate`**: Contains phrases indicating inability to calculate
- **`is_correct`**: `has_604` AND NOT `cannot_calculate`

**Interpretation**:
- **`is_correct = True`**: Answer is correct (contains $604 and no error messages)
- **`is_correct = False`**: Answer is incorrect or incomplete

---

## RAGAS Metrics (Optional)

When the `ragas` package is installed, the framework can also compute standard RAGAS metrics:

### 1. Faithfulness

**Purpose**: Measures how grounded the answer is in the provided context.

**What it measures**: Whether the answer can be inferred from the retrieved contexts without hallucination.

**Range**: 0.0 to 1.0 (higher is better)

**Use Case**: Detects when the model generates information not present in the retrieved documents.

---

### 2. Answer Relevancy

**Purpose**: Evaluates how relevant the answer is to the question.

**What it measures**: Semantic similarity between the question and the generated answer.

**Range**: 0.0 to 1.0 (higher is better)

**Use Case**: Detects when answers are off-topic or don't address the question.

---

### 3. Context Precision

**Purpose**: Measures how many of the retrieved contexts are relevant to answering the question.

**What it measures**: Proportion of retrieved contexts that are actually useful for answering the question.

**Range**: 0.0 to 1.0 (higher is better)

**Use Case**: Evaluates retrieval quality - are we retrieving the right documents?

---

### 4. Context Recall

**Purpose**: Measures how much of the relevant information from the ground truth is present in the retrieved contexts.

**What it measures**: Coverage of ground truth information in the retrieved contexts.

**Range**: 0.0 to 1.0 (higher is better)

**Use Case**: Evaluates retrieval completeness - did we retrieve all necessary information?

---

## Metric Summary Table

| Metric | Type | Range | Best For |
|--------|------|-------|----------|
| Exact Match | Boolean | True/False | Precise answers, calculations |
| Keyword Coverage | Percentage | 0.0 - 1.0 | List questions, descriptive answers |
| Number Match | Percentage | 0.0 - 1.0 | Calculations, numeric answers |
| Answer Completeness | Percentage | 0.0 - 1.0 | Comprehensive answers |
| Source Citation | Percentage | 0.0 - 1.0 | Traceability, verification |
| Rules Coverage (EF_1) | Percentage | 0.0 - 1.0 | List completeness |
| Fact Score (EF_2) | Percentage | 0.0 - 1.0 | Factual accuracy |
| Is Correct (EF_3) | Boolean | True/False | Calculation correctness |
| Faithfulness (RAGAS) | Score | 0.0 - 1.0 | Hallucination detection |
| Answer Relevancy (RAGAS) | Score | 0.0 - 1.0 | Answer relevance |
| Context Precision (RAGAS) | Score | 0.0 - 1.0 | Retrieval quality |
| Context Recall (RAGAS) | Score | 0.0 - 1.0 | Retrieval completeness |

---

## Interpreting Results

### High Scores Across All Metrics
- **Excellent**: The system is performing well across all dimensions.

### High Exact Match, Low Keyword Coverage
- **Possible Issue**: Answer is correct but missing important details or context.

### Low Exact Match, High Keyword Coverage
- **Possible Issue**: Answer contains the right information but is phrased differently or includes extra content.

### High Keyword Coverage, Low Number Match
- **Possible Issue**: Answer mentions the right concepts but has incorrect numeric values (critical for calculations).

### Low Source Citation
- **Possible Issue**: Answers lack traceability - difficult to verify or debug.

### High Custom Metrics, Low RAGAS Metrics
- **Possible Issue**: Answers match expected output but may not be well-grounded in retrieved contexts (potential overfitting to expected format).

---

## Best Practices

1. **Use Multiple Metrics**: No single metric captures everything. Use a combination for comprehensive evaluation.

2. **Question-Specific Evaluation**: Leverage question-specific metrics (EF_1, EF_2, EF_3) for targeted assessment.

3. **Context Matters**: For RAGAS metrics, ensure contexts are properly extracted and provided.

4. **Baseline Comparison**: Compare metrics against a baseline (e.g., previous version, different retrieval strategy).

5. **Error Analysis**: When metrics are low, examine specific examples to understand failure modes.

---

## Implementation Notes

- **Fast Mode**: The evaluation can run in "fast mode" using cached results, skipping context extraction for speed.
- **RAGAS Optional**: RAGAS metrics require the `ragas` package and are optional. They require context extraction, which is slower.
- **Question-Specific Logic**: The framework automatically applies question-specific checks based on question IDs (EF_1, EF_2, EF_3).

---

## References

- **RAGAS**: [RAGAS Documentation](https://docs.ragas.io/)
- **Evaluation Framework**: See `src/evaluation/evaluate.py` for implementation details

