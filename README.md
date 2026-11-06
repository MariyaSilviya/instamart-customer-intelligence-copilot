# Instamart Customer Intelligence & Support Copilot

## Overview

This project develops an end-to-end customer intelligence and support automation system using a deterministic synthetic Instamart dataset. It combines data cleaning, customer segmentation, lapse prediction, support ticket triage, policy retrieval, and a tool-using refund assistant.

The project is implemented in Python using pandas, NumPy, scikit-learn, SQLite, TF-IDF retrieval, and the supplied offline `mock_llm.py`. No paid APIs or external LLM services are required.

The project contains four parts:

- **Part A – Data Foundation:** Audits and cleans raw customer, product, order, and ticket data; stores trusted tables in SQLite; and validates analytical SQL queries.
- **Part B – Customer Intelligence:** Builds RFM features, performs K-means clustering and PCA, and trains models to predict customer lapse.
- **Part C – Support Ticket Triage:** Masks personally identifiable information (PII), classifies tickets using zero-shot and few-shot prompts, and applies retry, token-budget, and human-review guardrails.
- **Part D – Policy RAG and Refund Agent:** Retrieves information from policy documents and implements a ReAct-style agent that looks up orders, checks refund eligibility, issues simulated refunds, or escalates cases.

All monetary values are represented in INR.

## Reproduce

### Requirements

- Python 3.11 or later
- SQLite 3.25 or later
- Packages listed in `requirements.txt`
- Windows, macOS, or Linux

### Setup and execution

Run the following commands from the repository root.

1. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

2. Activate the environment.

   **Windows Command Prompt:**

   ```cmd
   venv\Scripts\activate
   ```

   **macOS/Linux:**

   ```bash
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Generate the deterministic synthetic dataset:

   ```bash
   python generate_data.py
   ```

5. Run all four project stages:

   ```bash
   python run_project.py
   ```

   The runner executes:

   - `part_a_pipeline.py`
   - `part_b_pipeline.py`
   - `part_c_pipeline.py`
   - `part_d_pipeline.py`

   If the raw data directory is empty, `run_project.py` automatically runs `generate_data.py`.

6. Run the additional Part A SQL and validation checks:

   ```bash
   python test_queries.py
   python validate_part_a.py
   ```

The main database is stored at `data/instamart.db`. Generated results are written to `outputs/`, and agent checkpoint files are stored in `state/`.

To run an individual stage after generating the data, use:

```bash
python part_a_pipeline.py
python part_b_pipeline.py
python part_c_pipeline.py
python part_d_pipeline.py
```

## Part A – Data Foundation: Cleaning, SQLite and SQL Analysis

### Data audit

The generated raw dataset contains:

| Dataset | Raw records |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Orders | 13,084 |
| Tickets | 164 |

The raw data was audited for duplicate records, inconsistent categories, missing values, invalid monetary values, delivery-time anomalies, and order-status inconsistencies.

Important audit findings included:

- 119 exact duplicate order rows.
- 656 pending order records.
- 120 non-numeric order amounts.
- 106 negative delivery-time values.
- 2,541 blank delivery-time values.
- 61 missing customer ages and 6 ages outside the accepted 10–100 range.
- 30 blank gender values.

### Cleaning and transformation

The cleaning pipeline applies the following rules:

- Standardizes city and gender values, using `unknown` for missing gender.
- Converts customer signup dates into datetime values.
- Replaces missing or implausible ages using city-level median ages.
- Removes exact duplicate order records.
- Retains the latest record for each `order_id` when multiple status records exist.
- Converts order amounts to numeric values and imputes invalid amounts using product price, quantity, and discount percentage.
- Converts negative delivery times to missing values.
- Imputes missing delivery times for completed orders using product-category medians calculated from completed orders.
- Retains missing delivery times for cancelled orders because those orders do not have a completed delivery.

### Cleaning results

| Metric | Result |
|---|---:|
| Clean orders | 12,309 |
| Completed orders | 11,110 |
| Cancelled orders | 1,199 |
| Pending orders remaining | 0 |
| Completed orders with missing delivery time | 0 |
| Cancelled orders with missing delivery time | 1,199 |
| Missing cleaned order amounts | 0 |
| Completed order amount total | ₹2,998,668.80 |

The clean database contains three analytical tables:

| SQLite table | Rows |
|---|---:|
| `customers` | 1,200 |
| `products` | 40 |
| `orders` | 12,309 |

### Outlier and correlation analysis

The order amount distribution is right-skewed:

- Mean order amount: ₹269.91
- Median order amount: ₹135.00
- Skewness: 4.61
- First quartile: ₹64.80
- Third quartile: ₹297.00
- Upper IQR fence: ₹645.30
- Observed upper outliers: 897

The strongest absolute correlation among the examined order amount relationships was with discount percentage (approximately -0.095). This is a weak negative relationship and does not establish causation.

### SQL validation

Seven analytical SQL queries were executed and cross-checked using pandas. The checks covered city-level revenue, delivery performance, monthly revenue, and order-status distribution.

Selected results:

- Completed orders: 90.26% of cleaned orders.
- Cancelled orders: 9.74%.
- Highest completed revenue by city: Hyderabad, ₹528,589.90.
- Fastest average delivery among the reported cities: Bengaluru, 27 minutes.

The SQL statements are available in `sql/queries.sql`.

## Part B – Customer Segmentation and Lapse Prediction

### Feature engineering

Customer-level RFM features were derived from completed order history:

- **Recency:** Days since the customer's most recent completed order.
- **Frequency:** Number of completed orders.
- **Monetary:** Total completed order value.

The resulting RFM dataset contains 1,175 customers.

| Metric | Result |
|---|---:|
| Mean frequency | 9.46 |
| Mean monetary value | ₹2,552.06 |
| Median recency | 19 days |
| Minimum recency | 0 days |
| Maximum recency | 180 days |

### K-means segmentation

Features were scaled before clustering. K-means models were evaluated for K values from 2 to 8 using inertia and silhouette score.

| K | Inertia | Silhouette score |
|---:|---:|---:|
| 2 | 1820.76 | 0.465 |
| 3 | 1268.23 | 0.445 |
| 4 | 743.07 | 0.526 |
| 5 | 617.37 | 0.480 |
| 6 | 503.17 | 0.482 |
| 7 | 408.46 | 0.416 |
| 8 | 366.50 | 0.412 |

K=4 was selected because it achieved the highest silhouette score among the evaluated configurations.

The resulting segments were profiled using recency, frequency, and monetary value. Segment labels are descriptive interpretations of cluster profiles rather than predefined customer categories.

PCA explained approximately 85.23% of the variance through the first two components:

- PC1: 63.43%
- PC2: 21.80%

### Lapse prediction

A lapse classification dataset was prepared with 1,156 customers, including 443 lapsed customers (38.32%).

A stratified train/test split produced:

- Training records: 924
- Testing records: 232

A baseline model achieved 61.64% accuracy but an F1 score of zero, illustrating why accuracy alone is insufficient for an imbalanced classification problem.

The evaluated models included Logistic Regression and Random Forest with hyperparameter tuning.

| Model | Test precision | Test recall | Test F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.6545 | 0.8090 | 0.7236 |
| Random Forest | 0.6735 | 0.7416 | 0.7059 |

Logistic Regression was selected based on its higher test F1 score of 0.7236.

Random Forest feature importance analysis identified frequency, monetary value, and recency as the leading features. The oversampling experiment increased recall to 0.9101, with an F1 score of 0.7105 and precision of 0.5827. This illustrates the trade-off between identifying more potentially lapsed customers and increasing false positives.

The selected model was saved and reloaded, and prediction equality was verified after reloading.

### Outputs

Key Part B outputs include:

- `customer_segments.csv`
- `segment_profiles.csv`
- `kmeans_evaluation.csv`
- `rfm_table.csv`
- `lapse_model.joblib`
- `lapse_model_metrics.csv`
- `retention_list.csv`
- `oversampling_comparison.csv`
- `random_forest_feature_importance.csv`
- `logistic_regression_coefficients.csv`
- `pca_coordinates.csv`

The retention list contains 10 customers ranked by predicted lapse probability. These predictions are prioritization signals, not confirmed churn events.

## Part C – Support Ticket Triage and Privacy Guardrails

### Privacy masking

The pipeline processes 164 support tickets and masks sensitive information before sending text to the mock LLM.

The masking process recorded:

| PII pattern | Substitutions |
|---|---:|
| Email addresses | 55 |
| Phone numbers | 91 |
| Order IDs | 132 |
| Total | 278 |

A post-masking validation confirmed that the targeted patterns were no longer present in the processed text.

### Input handling

The ticket dataset contains:

- 4 empty tickets.
- 4 oversized tickets.
- 156 tickets eligible for normal classification.

Empty and oversized inputs are handled separately instead of being passed through the normal classification flow.

### Zero-shot and few-shot classification

The supplied `mock_llm.py` is used in offline classification mode. The pipeline evaluates zero-shot and few-shot prompting.

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Accuracy | 78.75% | 91.875% |
| Token usage | 9,822 | 14,897 |
| Retries | 19 | 19 |

Accuracy was evaluated across 160 labelled records. Few-shot prompting improved accuracy by 13.125 percentage points, with higher token usage.

Ticket categories include billing, delivery, technical, and fraud. Invalid model responses are handled through fallback and review logic.

### Guardrails and outputs

The pipeline includes:

- PII masking before LLM processing.
- Retry handling for transient errors.
- Invalid-output handling.
- Empty and oversized ticket handling.
- Token-budget enforcement.
- Human-review routing for cases requiring additional attention.

The budget demonstration used a 300-token budget and consumed 255 tokens. Three tickets were classified before the remaining eligible requests were marked as budget-exhausted.

The human-review output contains 17 records. The fraud security output contains 34 fraud tickets, all assigned high priority and routed to the Security Team.

Key outputs include:

- `triage_results.csv`
- `triage_zero_shot_results.csv`
- `triage_fewshot_comparison.csv`
- `triage_budget_demo.csv`
- `triage_human_review.csv`
- `triage_fraud_security.csv`
- `triage_retry.log`
- `zero_shot_confusion_matrix.csv`
- `few_shot_confusion_matrix.csv`

## Part D – Policy RAG Assistant and Refund Agent

### Policy document retrieval

Five policy documents are indexed:

- `refund_policy.txt`
- `delivery_policy.txt`
- `payment_policy.txt`
- `privacy_policy.txt`
- `escalation_policy.txt`

The retrieval pipeline uses TF-IDF with 40-word chunks and 10-word overlap, producing 15 chunks.

The evaluation contains 14 questions:

- 12 questions with answers grounded in policy documents.
- 2 questions about information not available in the policy documents.

With overlapping chunks, all 12 answerable questions matched their expected facts, and both unanswerable questions received an abstention response indicating that the information was unavailable.

A zero-overlap comparison produced 13 chunks and recorded 12 fact matches out of 14 total evaluation cases, compared with 14 out of 14 for the overlapping configuration. The evaluation treats the expected abstention on unanswerable questions as a successful outcome; its retrieval-match field should therefore not be interpreted as proof that a relevant document exists for every question.

Some generated answers contain repeated or incomplete phrases because overlapping chunks can contain repeated text. This is a retrieval and answer-quality limitation.

### Refund agent

The ReAct-style agent uses five tools:

1. `get_order` – retrieves order information.
2. `search_policy` – retrieves relevant refund policy information.
3. `calculate_refund` – calculates the eligible refund.
4. `issue_refund` – records a simulated refund.
5. `escalate_to_human` – routes cases requiring human approval.

The agent follows the refund policy:

- Only completed orders are eligible for refunds.
- Orders delivered in more than 45 minutes qualify for a full refund.
- Orders delivered in 45 minutes or less do not qualify for a late-delivery refund.
- Refunds up to ₹1,000 can be approved automatically.
- Refunds above ₹1,000 require human support lead approval.

### Agent scenario evaluation

| Scenario | Expected outcome | Verified result |
|---|---|---|
| S1 | Refund ₹128.25 | Passed |
| S2 | Refund ₹135 | Passed |
| S3 | Escalate ₹1,999 | Passed |
| S4 | Deny – delivery within 45 minutes | Passed |
| S5 | Deny – order not completed | Passed |
| S6 | Order not found | Passed |
| S7 | Maximum-iteration fallback | Passed |

All seven scenario outcomes passed validation.

The refund ledger contains two successful simulated refund entries:

- S1: ₹128.25
- S2: ₹135.00
- Total: ₹263.25

Denied, not-found, and escalation scenarios did not create refund ledger entries.

### Reliability demonstrations

The pipeline also demonstrates:

- Checkpoint creation and resume for S1.
- Restoration of the saved conversation and tool observations.
- Verification that each S1 tool was called once.
- Token-limit fallback.
- Timeout fallback.
- Maximum-iteration fallback.
- A ReAct versus Plan-Execute workflow comparison.

The token-limit and timeout cases are controlled demonstrations using restrictive test thresholds, not measurements of production operating limits.

Key outputs include:

- `policy_chunks.csv`
- `policy_rag_results.csv`
- `policy_rag_comparison.csv`
- `policy_rag_zero_overlap_results.csv`
- `agent_scenario_results.csv`
- `agent_runs.jsonl`
- `refund_ledger.jsonl`
- `react_vs_plan_execute.md`

Checkpoint files are stored in `state/`.

## Limitations and Next Steps

### Limitations

- The dataset is synthetic and deterministically generated. Results may not represent real customer behaviour.
- The supplied `mock_llm.py` is an offline deterministic simulation, not a live LLM integration.
- Refund execution is simulated. No real payment gateway or customer account is connected.
- The lapse model is trained and evaluated on synthetic data; its performance must not be assumed to generalize to real customers.
- TF-IDF retrieval is a lightweight baseline and can return overlapping or repetitive text.
- The ticket classification and refund-agent workflows require human oversight for uncertain, invalid, high-value, or exceptional cases.
- Token, timeout, and iteration fallback demonstrations use controlled test conditions.

### Next steps

- Evaluate the pipelines on anonymized real-world data, subject to appropriate privacy and security controls.
- Improve policy retrieval with stronger chunking, reranking, and answer-grounding checks.
- Add more robust evaluation for unanswerable questions and hallucination detection.
- Integrate a production LLM only with appropriate access controls, monitoring, rate limits, and cost controls.
- Connect refund workflows to approved payment services with authentication, authorization, audit logs, and human approval.
- Deploy ticket triage through a monitored support integration and measure performance continuously.
