# Zesty RAG Assignment

A production-ready RAG (Retrieval-Augmented Generation) system for analyzing insurance regulatory PDFs with mixed structured and unstructured content.

## Overview

This system implements a hybrid RAG architecture capable of answering complex questions requiring both semantic understanding and precise quantitative data extraction from structured tables. It processes 40+ PDF documents and handles three types of queries:

1. **List Queries**: Comprehensive enumeration (e.g., "List all rating plan rules")
2. **Comparison Queries**: Geographic reasoning with territory-based comparisons
3. **Calculation Queries**: Multi-step calculations requiring values from multiple tables

## Results

### Evaluation Summary

| Question | Type | Keyword Coverage | Number Match | Completeness | Source Citation | Special Metric |
|----------|------|------------------|--------------|--------------|-----------------|----------------|
| **EF_1** | List | 73.7% | 100.0% | 86.8% | 100.0% | Rules Coverage: 60.6% (20/33) |
| **EF_2** | Comparison | 90.0% | 77.8% | 95.0% | 100.0% | Fact Score: 100.0% |
| **EF_3** | Calculation | 100.0% | 100.0% | 100.0% | 100.0% | Calculation Correct: ✅ |

**Overall Performance:**
- Average Keyword Coverage: **87.9%**
- Average Number Match: **92.6%**
- Average Answer Completeness: **93.9%**
- Average Source Citation Quality: **100.0%**

### Question 1: EF_1 - List All Rating Plan Rules

**Question:**
> List all rating plan rules

**Generated Answer:**
```
Based on the provided documents, the following is a comprehensive list of all rating plan rules for the MAPS Homeowner program. The rating plan is primarily detailed in "PART C—RATING PLAN".

### Rating Plan Rules (Section C)

A table of contents in the "CT MAPS Homeowner Rules Manual" lists the following rules under "SECTION C —- RATING PLAN":
(SOURCE: (215066178-180449588)-CT MAPS Homeowner Rules Manual eff 08.18.25 v4.pdf, page 2, TABLE: (215066178-180449588)-CT MAPS Homeowner Rules Manual eff 08.18.25 v4_table_0)

*   **Rule C-1:** Limits of Liability and Coverage Relationships
*   **Rule C-2:** Rating Perils
*   **Rule C-3:** Base Rates
*   **Rule C-4:** Policy Type Factor
*   **Rule C-5:** Policy Tier Guidelines
*   **Rule C-6:** Amount of Insurance/Deductibles
*   **Rule C-7:** Hurricane Deductibles
*   **Rule C-8:** Windstorm/Hail Deductibles
*   **Rule C-9:** Policy Territory Determination
*   **Rule C-10:** Distance to Coast Factor
*   **Rule C-11:** Public Protection Class Factors
*   **Rule C-12:** Age of Home Factor
*   **Rule C-13:** Year Built Factor
*   **Rule C-14:** Account Discount
*   **Rule C-15:** Roof Type Factor
*   **Rule C-16:** Dwelling Usage Factor
*   **Rule C-17:** Increased Limits
*   **Rule C-18:** Protective Device Discount
*   **Rule C-19:** Affinity Discount
*   **Rule C-20:** Association Discount
*   **Rule C-21:** Oil Tank Factor
*   **Rule C-22:** Pool Factor

### Additional Rating Plan Rules

Other documents mention additional rules that are part of the rating plan but are not listed in the provided table of contents view:

*   **Rule C-24: Roof Condition Factor:** This rule's factors are based on the Roof Condition Score from Cape Analytics. (SOURCE: (summary_document_AMEC-134363726_2025-06-06)-AMEC-134363726.pdf, page 8)
*   **Rule C-32: Loss History Rating:** This plan recognizes the loss history of an insured for both property and liability coverages to determine the premium for new and renewal policies. (SOURCE: (215066178-180449588)-CT MAPS Homeowner Rules Manual eff 08.18.25 v4.pdf, page 50; (215066178-180449602)-CT Legacy Homeowner Rules eff 04.01.24 mu to MAPS Homeowner Rules eff 8.18.25 v3.pdf, page 15)
```

**Evaluation Metrics:**
- **Keyword Coverage**: 73.7%
- **Number Match**: 100.0%
- **Completeness**: 86.8%
- **Source Citation**: 100.0%
- **Rules Coverage**: 60.6% (20 out of 33 expected rules found)

**Expected Output:** 33 rules (see `artifacts/questions.csv` for complete list)

---

### Question 2: EF_2 - Multi-Step Comparison Question

**Question:**
> For a State Farm Fire & Casualty Company policyholder living in Zip Code 89110, does the resident living West of Nellis Boulevard face a higher proposed percentage rate change for Comprehensive coverage compared to a resident living East of Nellis Boulevard, and does the numeric difference between the Collision Rating Groups (GRG) of a 2023 Ducati Panigale V4 R and a 2023 Honda Grom ABS exceed 30 points?

**Generated Answer:**
```
Based on the information provided, here is the analysis for both parts of your question:

### Part 1: Comprehensive Coverage Rate Change Comparison

Yes, a resident living West of Nellis Boulevard in ZIP code 89110 faces a higher proposed percentage rate change for Comprehensive coverage compared to a resident living East of Nellis Boulevard.

**1. Territory Identification:**
*   **West of Nellis Boulevard (Territory 118):** The document `MCY Rate Filing Data Summary - SFFC.pdf` defines the area including "The portion of ZIP Code 89110 west of Nellis Boulevard" as part of Territory 118 (page 21).
*   **East of Nellis Boulevard (Territory 117):** The remainder of ZIP code 89110, which would be East of Nellis Boulevard, falls into Territory 117. This territory is described as the portions of ZIP codes 89030, 89110, and 89115 not included in Territory 118 (page 20, `MCY Rate Filing Data Summary - SFM.pdf`, table `(213128717-179157012)-MCY Rate Filing Data Summary - SFM_table_22`).

**2. Proposed Rate Changes:**
The proposed rate changes for State Farm Fire & Casualty Company are found in the `MCY Rate Filing Data Summary - SFFC.pdf` document (page 2, table `(213128717-179157013)-MCY Rate Filing Data Summary - SFFC_table_1`):
*   **Territory 118 (West of Nellis):** The proposed Comprehensive coverage change is **+0.305%**.
*   **Territory 117 (East of Nellis):** The proposed Comprehensive coverage change is **-0.133%**.

Therefore, the resident West of Nellis Boulevard has a higher proposed rate change.

---

### Part 2: Motorcycle Collision Rating Group (GRG) Difference

Yes, the numeric difference between the Collision Rating Groups (GRG) of a 2023 Ducati Panigale V4 R and a 2023 Honda Grom ABS exceeds 30 points.

**1. GRG Values:**
While the question specifies 2023 models, the necessary values are located in the rate manual pages for 2024 models. The filing notes that it includes updates for both 2017-2023 and additional 2024 models.

*   **2023 Ducati Panigale V4 R:** The GRG is **51**.
    *   Source: `2024 CW Rate Manual Pages.pdf`, page 11, table `(213128625-179157161)-2024 CW Rate Manual Pages_table_10`
*   **2023 Honda Grom ABS:** The GRG is **15**.
    *   Source: `2024 CW Rate Manual Pages.pdf`, page 3, table `(213128625-179157161)-2024 CW Rate Manual Pages_table_2`

**2. Calculation:**
*   Difference = GRG of Ducati Panigale V4 R - GRG of Honda Grom ABS
*   Difference = 51 - 15 = **36**

The difference of 36 points is greater than 30.
```

**Evaluation Metrics:**
- **Keyword Coverage**: 90.0%
- **Number Match**: 77.8%
- **Completeness**: 95.0%
- **Source Citation**: 100.0%
- **Fact Score**: 100.0% (all 8 facts present: Yes, Territory 118, Territory 117, 0.305%, -0.133%, GRG 51, GRG 15, difference 36)

**Expected Output:** "Yes. A resident living West of Nellis Boulevard (Territory 118) faces a higher proposed Comprehensive rate change (0.305%) than a resident living East of Nellis Boulevard (Territory 117, -0.133%). Additionally, the difference between the Collision Rating Group of the Ducati Panigale V4 R (051) and the Honda Grom ABS (015) is 36, which exceeds 30 points."

---

### Question 3: EF_3 - Calculate Hurricane Premium

**Question:**
> Using the Base Rate and the applicable Mandatory Hurricane Deductible Factor, calculate the unadjusted Hurricane premium for an HO3 policy with a $750,000 Coverage A limit located 3,000 feet from the coast in a Coastline Neighborhood.

**Generated Answer:**
```
To calculate the unadjusted Hurricane premium, we need to find the Hurricane Base Rate and the applicable Mandatory Hurricane Deductible Factor.

**1. Find the Hurricane Base Rate:**

*   According to the "Base Rates" table from the CT Homeowners MAPS Rate Pages, the Hurricane Base Rate is a fixed value.
*   In table `(215004905-180407973)-CT Homeowners MAPS Rate Pages Eff 8.18.25 v3_table_5` on page 4 of the "CT Homeowners MAPS Rate Pages Eff 8.18.25 v3.pdf", the value in the **Hurricane** column is **$293**.

**2. Determine the Applicable Hurricane Deductible Percentage:**

*   The policy is for a property located 3,000 feet from the coast.
*   According to the rules in "CT MAPS Homeowner Rules Manual eff 08.18.25 v4.pdf" on page 23, "Properties located in the State designated "Coastline Neighborhoods" that are greater than > 2500 from the coast will receive a 2% deductible."
*   Since 3,000 feet is greater than 2,500 feet, the applicable mandatory hurricane deductible is **2%**.

**3. Find the Mandatory Hurricane Deductible Factor:**

*   This factor is found in the "Hurricane Deductible Factor" table, which varies by Coverage A limit and the deductible percentage.
*   We will use table `(215004905-180407973)-CT Homeowners MAPS Rate Pages Eff 8.18.25 v3_table_119` on page 72 of the "CT Homeowners MAPS Rate Pages Eff 8.18.25 v3.pdf".
*   In this table, we look for the row corresponding to:
    *   **Policy Form:** HO3
    *   **Coverage A Limit:** $750,000
    *   **Applicable Hurricane Deductible:** 2%
*   The value in the **Hurricane** column for this row is **2.061**.

**4. Calculate the Unadjusted Hurricane Premium:**

*   **Formula:** Unadjusted Hurricane Premium = Hurricane Base Rate × Mandatory Hurricane Deductible Factor
*   **Calculation:** $293 × 2.061 = $603.873
*   **Rounding:** The final premium is rounded to the nearest dollar.

$603.873 rounds to **$604**.

**Final Answer:**

Based on the provided documents, the unadjusted Hurricane premium is calculated as follows:

*   **Hurricane Base Rate:** $293 (Source: CT Homeowners MAPS Rate Pages Eff 8.18.25 v3.pdf, page 4, table `(215004905-180407973)-CT Homeowners MAPS Rate Pages Eff 8.18.25 v3_table_5`)
*   **Mandatory Hurricane Deductible Factor:** 2.061 (Source: CT Homeowners MAPS Rate Pages Eff 8.18.25 v3.pdf, page 72, table `(215004905-180407973)-CT Homeowners MAPS Rate Pages Eff 8.18.25 v3_table_119`)
*   **Calculation:** $293 * 2.061 = $603.873

Rounding to the nearest dollar, the unadjusted Hurricane premium is **$604**.
```

**Evaluation Metrics:**
- **Keyword Coverage**: 100.0%
- **Number Match**: 100.0%
- **Completeness**: 100.0%
- **Source Citation**: 100.0%
- **Calculation Correct**: ✅ (contains $604, $293, 2.061, shows calculation steps)

**Expected Output:** $604

---

### Detailed Evaluation Metrics

For a complete breakdown of evaluation metrics, see:
- [Evaluation Report](results_evaluation/evaluation_report.txt)
- [Evaluation Results CSV](results_evaluation/evaluation_results.csv)
- [Evaluation Metrics Documentation](results_evaluation/EVALUATION_METRICS.md)

## Architecture

### Dual-Index Strategy
- **DOC Index**: BigQuery Vector Store for semantic retrieval of text chunks and table summaries
- **FACTS Store**: BigQuery tables for precise structured data queries

### Hybrid Retrieval
- Semantic search (vector similarity)
- Value-based table search
- Column-based table search
- Full table retrieval (instead of SQL generation)

### Key Features
- Handles inconsistent table schemas (generic headers, missing headers)
- Detects and merges tables split across pages
- Dynamic table discovery (no hardcoded table names)
- Page/document-level expansion for comprehensive context
- Parallel processing for performance
- Comprehensive evaluation framework

## Project Structure

```
.
├── src/
│   ├── parsing/              # PDF parsing and extraction
│   │   ├── pdf_parser.py     # LlamaParse + Unstructured integration
│   │   ├── table_processor.py  # Table normalization and LLM summarization
│   │   └── README.md         # Parsing module documentation
│   ├── indexing/            # Index creation (vector store, BigQuery)
│   │   ├── vector_store.py   # DOC Index builder (vector embeddings)
│   │   ├── facts_store.py    # FACTS Store builder (BigQuery tables)
│   │   └── README.md         # Indexing module documentation
│   ├── retrieval/           # RAG agent and retrieval logic
│   │   ├── rag_agent.py      # Main RAG agent orchestrator
│   │   ├── retrieval_logic.py # Core retrieval algorithms
│   │   ├── utils.py          # Utility functions
│   │   └── prompt_template_v2.txt  # LLM prompt template
│   └── evaluation/          # Evaluation framework
│       ├── evaluate.py       # Comprehensive evaluation metrics
│       └── __init__.py
├── artifacts/               # Input data and questions
│   ├── 1/                   # PDF folder 1 (motorcycle filings)
│   ├── 2/                   # PDF folder 2 (homeowner filings)
│   ├── questions.csv        # Evaluation questions and expected outputs
│   └── README.md            # Human reasoning steps for questions
├── results_evaluation/      # Evaluation results and metrics
│   ├── questions_results.csv      # Generated answers
│   ├── evaluation_results.csv     # Detailed metrics
│   ├── evaluation_report.txt      # Summary report
│   └── EVALUATION_METRICS.md      # Metrics documentation
├── GenAI_Prompts/           # Architecture and design prompts
├── build_indices.py         # Main indexing pipeline script
├── query_rag.py             # RAG query script (interactive/batch)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## For Reviewers

### Quick Start Guide

If you're reviewing this project, follow these steps to get started:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ovsilya/zesty-assignment.git
   cd zesty-assignment
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up authentication:**
   - **GCP Access**: You'll need the service account key file (`zesty-481316-f5efb1098d47.json`) shared by the project owner
     - Place it in the project root directory (it will be auto-detected)
     - Or set `GOOGLE_APPLICATION_CREDENTIALS` in your `.env` file
   - **LlamaParse API Key**: Get a free API key from [LlamaCloud](https://cloud.llamaindex.ai/)

4. **Create `.env` file:**
   ```bash
   cp .env.example .env
   # Edit .env and add:
   # GOOGLE_CLOUD_PROJECT=zesty-481316
   # LLAMA_CLOUD_API_KEY=your-llamaparse-api-key
   ```

5. **Run the system:**
   ```bash
   # If indices are already built, you can query directly:
   python3 query_rag.py "List all rating plan rules"
   
   # Or build indices from PDFs (requires PDFs in artifacts/1/ and artifacts/2/):
   python3 build_indices.py
   ```

### What You Need

- **Service Account Key**: Provided by project owner (for GCP access)
- **LlamaParse API Key**: Free tier available at [LlamaCloud](https://cloud.llamaindex.ai/)
- **PDF Documents**: Not included in repository (place in `artifacts/1/` and `artifacts/2/` if building indices)

### Notes

- The service account key file (`zesty-*.json`) is automatically detected if placed in the project root
- All sensitive files (`.env`, `*.json`) are excluded from git
- The system automatically verifies authentication before running

## Usage

### Build Indices

```bash
python3 build_indices.py
```

### Query RAG System

```bash
# Interactive mode
python3 query_rag.py

# Single query
python3 query_rag.py "Your question here"

# Batch queries from CSV
python3 query_rag.py --file artifacts/questions.csv
```

### Evaluation

```bash
# Run evaluation on existing results
python3 src/evaluation/evaluate.py
```

## Development Journey

See [RAG Development Journey](GenAI_Prompts/RAG_DEVELOPMENT_JOURNEY.md) for detailed documentation of:
- Thought process and reasoning
- Development phases
- Key technical decisions
- Lessons learned

## Documentation

### Core Modules

- **[PDF Parsing](src/parsing/README.md)** - LlamaParse and Unstructured integration for extracting text, tables, and charts from PDFs
- **[Indexing](src/indexing/README.md)** - Building DOC Index (vector store) and FACTS Store (BigQuery tables)
- **[Retrieval](src/retrieval/)** - RAG agent with hybrid retrieval (semantic + structured)
- **[Evaluation](src/evaluation/evaluate.py)** - Comprehensive evaluation framework with custom and RAGAS metrics

### Architecture and Design

- **[RAG Development Journey](GenAI_Prompts/RAG_DEVELOPMENT_JOURNEY.md)** - Complete development story and decision-making process
- **[Unified Architecture](GenAI_Prompts/unified_architecture.md)** - System architecture design
- **[Engineering the Indices](GenAI_Prompts/Engineering%20the%20Indices%20–%20Design%20&%20Th.md)** - Detailed design document for index creation

### Execution and Evaluation

- **[Script Execution Flow](SCRIPT_EXECUTION_FLOW.md)** - Detailed script execution flow and dependencies
- **[Evaluation Metrics](results_evaluation/EVALUATION_METRICS.md)** - Explanation of all evaluation metrics used
