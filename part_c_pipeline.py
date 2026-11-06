# import os
# import re
# import json
# import pandas as pd
# from mock_llm import chat, TransientLLMError, reset

# os.makedirs("outputs", exist_ok=True)
# reset()

# print("--- 1. Loading Tickets and Applying Privacy Masking ---")
# tickets_df = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)

# def mask_pii(text):
#     # Mask email addresses
#     text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
#     # Mask phone numbers (10-digit numbers or similar patterns)
#     text = re.sub(r'\b\d{10}\b', '[PHONE]', text)
#     text = re.sub(r'\b\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
#     return text

# tickets_df["masked_text"] = tickets_df["text"].apply(mask_pii)

# print(f"Loaded and masked {len(tickets_df)} support tickets.")

# print("\n--- 2. Classifying Tickets using Mock LLM ---")
# results = []
# for idx, row in tickets_df.iterrows():
#     messages = [
#         {"role": "system", "content": "CLASSIFY: Categorize the support ticket into billing, delivery, technical, or fraud."},
#         {"role": "user", "content": row["masked_text"]}
#     ]
    
#     predicted_category = "unknown"
#     reason = ""
    
#     # Retry loop for handling TransientLLMError (simulated 503 server busy)
#     for attempt in range(3):
#         try:
#             response = chat(messages)
#             content = response.choices[0].message.content
            
#             # Try parsing JSON response from LLM
#             try:
#                 data = json.loads(content)
#                 predicted_category = data.get("category", "unknown")
#                 reason = data.get("reason", "")
#             except json.JSONDecodeError:
#                 # Handle non-JSON chatty answers
#                 content_lower = content.lower()
#                 for cat in ["fraud", "billing", "technical", "delivery"]:
#                     if cat in content_lower:
#                         predicted_category = cat
#                         break
#                 reason = "parsed from text response"
#             break
#         except TransientLLMError:
#             if attempt == 2:
#                 predicted_category = "technical"
#                 reason = "fallback after max retries"
#         except Exception:
#             predicted_category = "unknown"
#             reason = "error"
#             break

#     results.append({
#         "ticket_id": row["ticket_id"],
#         "customer_id": row["customer_id"],
#         "true_category": row["true_category"],
#         "predicted_category": predicted_category,
#         "reason": reason
#     })

# triage_df = pd.DataFrame(results)
# triage_df.to_csv("outputs/triage_results.csv", index=False)
# print("Triage results saved to outputs/triage_results.csv")

# # Generate budget demo output
# triage_df["budget_approved"] = triage_df["predicted_category"].isin(["billing", "delivery"])
# triage_df.to_csv("outputs/triage_budget_demo.csv", index=False)
# print("Budget demo saved to outputs/triage_budget_demo.csv")





import os
import re
import json
import time
import random
import logging

import numpy as np
import pandas as pd

from mock_llm import chat, TransientLLMError, reset, est_tokens
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# SETUP
# ============================================================

os.makedirs("outputs", exist_ok=True)

random.seed(42)

logging.basicConfig(
    filename="outputs/triage_retry.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filemode="w"
)

CATEGORIES = ["billing", "delivery", "technical", "fraud"]
MAX_TICKET_LENGTH = 500
MAX_RETRIES = 3
FINAL_TOKEN_BUDGET = 100000

print("=" * 65)
print("PART C: SUPPORT TRIAGE WITH PRIVACY AND BUDGET GUARDRAILS")
print("=" * 65)


# ============================================================
# TASK 1: LOAD TICKETS
# ============================================================

tickets = pd.read_csv(
    "data/raw/tickets.csv",
    keep_default_na=False
)

print("\n--- 1. Loading Tickets ---")
print("Total tickets:", len(tickets))

assert len(tickets) == 164


# ============================================================
# TASK 2: PII MASKING
# ============================================================

print("\n--- 2. Masking Personal Information ---")

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

PHONE_PATTERN = re.compile(
    r"\b\d{10}\b|\b\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)

ORDER_PATTERN = re.compile(r"\bORD-\d{6}\b", re.IGNORECASE)


def mask_pii(text):
    text = str(text)

    email_count = len(EMAIL_PATTERN.findall(text))
    phone_count = len(PHONE_PATTERN.findall(text))
    order_count = len(ORDER_PATTERN.findall(text))

    text = EMAIL_PATTERN.sub("[EMAIL]", text)
    text = PHONE_PATTERN.sub("[PHONE]", text)
    text = ORDER_PATTERN.sub("[ORDER_ID]", text)

    return text, email_count, phone_count, order_count


masked_values = tickets["text"].apply(mask_pii)

tickets["masked_text"] = masked_values.apply(lambda x: x[0])
tickets["email_count"] = masked_values.apply(lambda x: x[1])
tickets["phone_count"] = masked_values.apply(lambda x: x[2])
tickets["order_id_count"] = masked_values.apply(lambda x: x[3])

email_total = int(tickets["email_count"].sum())
phone_total = int(tickets["phone_count"].sum())
order_total = int(tickets["order_id_count"].sum())

print("Email substitutions:", email_total)
print("Phone substitutions:", phone_total)
print("Order ID substitutions:", order_total)
print("Total substitutions:", email_total + phone_total + order_total)

assert email_total == 55
assert phone_total == 91
assert order_total == 132
assert email_total + phone_total + order_total == 278

# Verify that no original PII patterns remain in masked ticket text.
for text in tickets["masked_text"]:
    assert not EMAIL_PATTERN.search(text)
    assert not PHONE_PATTERN.search(text)
    assert not ORDER_PATTERN.search(text)

print("PII verification passed: no matching original PII remains.")


# ============================================================
# TASK 3: EMPTY AND LONG TICKET HANDLING
# ============================================================

print("\n--- 3. Ticket Eligibility Checks ---")

tickets["text_length"] = tickets["masked_text"].str.len()

empty_mask = tickets["masked_text"].str.strip().eq("")
long_mask = tickets["text_length"] > MAX_TICKET_LENGTH

print("Empty tickets:", int(empty_mask.sum()))
print("Long tickets:", int(long_mask.sum()))

assert int(empty_mask.sum()) == 4
assert int(long_mask.sum()) == 4

eligible = tickets[~(empty_mask | long_mask)].copy()

print("Eligible tickets for LLM classification:", len(eligible))

assert len(eligible) == 156


# ============================================================
# TASK 4: CLASSIFICATION FUNCTION
# ============================================================

def classify_batch(data, few_shot=True, token_budget=100000, run_name="fewshot"):
    """
    Classify tickets using the supplied offline mock LLM.

    Empty and long tickets are skipped.
    Invalid responses and budget-exhausted tickets go to human review.
    Token usage is tracked and restricted by token_budget.
    """

    reset()

    results = []
    tokens_spent = 0
    retry_count = 0

    if few_shot:
        system_prompt = (
            "CLASSIFY: Categorize the support ticket into exactly one "
            "of billing, delivery, technical, or fraud. "
            "Return JSON with category and reason. "
            "EXAMPLES: "
            "A payment charged twice is billing. "
            "A stolen account is fraud. "
            "An application crash is technical. "
            "A late rider is delivery."
        )
    else:
        system_prompt = (
            "CLASSIFY: Categorize the support ticket into exactly one "
            "of billing, delivery, technical, or fraud. "
            "Return JSON with category and reason."
        )

    # Preserve all 164 tickets in every output.
    for _, row in tickets.iterrows():

        ticket_id = row["ticket_id"]
        text = row["masked_text"]

        result = {
            "ticket_id": ticket_id,
            "customer_id": row["customer_id"],
            "true_category": row["true_category"],
            "predicted_category": "human_review",
            "reason": "",
            "status": "",
            "priority": "NORMAL",
            "assigned_team": "Customer Support",
            "email_redactions": int(row["email_count"]),
            "phone_redactions": int(row["phone_count"]),
            "order_id_redactions": int(row["order_id_count"]),
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        # Empty ticket: no model call.
        if text.strip() == "":
            result["status"] = "skipped_empty"
            result["reason"] = "empty ticket sent for human review"
            results.append(result)
            continue

        # Long ticket: no model call.
        if len(text) > MAX_TICKET_LENGTH:
            result["status"] = "skipped_long"
            result["reason"] = "long ticket sent for human review"
            results.append(result)
            continue

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]

        successful_response = False
        budget_exhausted = False

        for attempt in range(1, MAX_RETRIES + 1):

            # Conservative reservation for prompt plus completion.
            estimated_prompt = sum(
                est_tokens(message.get("content", ""))
                for message in messages
            )

            reserved_tokens = estimated_prompt + 25

            if tokens_spent + reserved_tokens > token_budget:
                budget_exhausted = True
                break

            result["llm_calls"] += 1

            try:
                response = chat(messages)

                usage = response.usage

                prompt_tokens = int(usage.prompt_tokens)
                completion_tokens = int(usage.completion_tokens)
                total_tokens = int(usage.total_tokens)

                tokens_spent += total_tokens

                result["prompt_tokens"] += prompt_tokens
                result["completion_tokens"] += completion_tokens
                result["total_tokens"] += total_tokens

                content = response.choices[0].message.content

                # First attempt JSON parsing.
                try:
                    parsed = json.loads(content)

                    category = str(
                        parsed.get("category", "")
                    ).strip().lower()

                    if category not in CATEGORIES:
                        result["status"] = "flagged_invalid_output"
                        result["reason"] = (
                            "invalid category returned: " + category
                        )
                    else:
                        result["predicted_category"] = category
                        result["reason"] = parsed.get(
                            "reason", "JSON classification"
                        )
                        result["status"] = "classified_json"

                except (json.JSONDecodeError, AttributeError, TypeError):
                    # Try extracting a valid category from plain text.
                    content_lower = str(content).lower()

                    found_category = next(
                        (
                            category
                            for category in CATEGORIES
                            if category in content_lower
                        ),
                        None
                    )

                    if found_category:
                        result["predicted_category"] = found_category
                        result["reason"] = "parsed from text response"
                        result["status"] = "classified_keyword"
                    else:
                        result["status"] = "flagged_invalid_output"
                        result["reason"] = "unparseable model response"

                successful_response = True
                break

            except TransientLLMError as error:
                retry_count += 1

                # Failed attempts are conservatively charged using
                # the reserved estimate.
                tokens_spent += reserved_tokens

                logging.warning(
                    "%s | ticket=%s | attempt=%s | error=%s",
                    run_name,
                    ticket_id,
                    attempt,
                    str(error)
                )

                if attempt < MAX_RETRIES:
                    jitter = random.uniform(0.01, 0.05)
                    time.sleep(jitter)

            except Exception as error:
                result["status"] = "flagged_invalid_output"
                result["reason"] = "model error: " + str(error)
                logging.exception(
                    "%s | ticket=%s | unexpected error",
                    run_name,
                    ticket_id
                )
                break

        if budget_exhausted:
            result["status"] = "budget_exhausted"
            result["reason"] = "token budget insufficient; human review required"

        elif not successful_response and not result["status"]:
            result["status"] = "flagged_invalid_output"
            result["reason"] = "maximum retries reached"

        # Fraud cases go to the security team with HIGH priority.
        if result["predicted_category"] == "fraud":
            result["priority"] = "HIGH"
            result["assigned_team"] = "Security Team"

        # Invalid, skipped and budget-exhausted tickets require review.
        if result["status"] in [
            "flagged_invalid_output",
            "skipped_empty",
            "skipped_long",
            "budget_exhausted"
        ]:
            result["predicted_category"] = "human_review"

        results.append(result)

    output = pd.DataFrame(results)

    assert len(output) == 164

    return output, tokens_spent, retry_count


# ============================================================
# TASK 5: ZERO-SHOT EXPERIMENT
# ============================================================

print("\n--- 4. Zero-Shot Classification Experiment ---")

zero_shot_results, zero_tokens, zero_retries = classify_batch(
    eligible,
    few_shot=False,
    token_budget=FINAL_TOKEN_BUDGET,
    run_name="zero_shot"
)

zero_shot_results.to_csv(
    "outputs/triage_zero_shot_results.csv",
    index=False
)

print("Zero-shot rows:", len(zero_shot_results))
print("Zero-shot token usage:", zero_tokens)
print("Zero-shot retries:", zero_retries)
print(
    "Zero-shot statuses:\n",
    zero_shot_results["status"].value_counts().to_string()
)


# ============================================================
# TASK 6: FEW-SHOT FINAL CLASSIFICATION
# ============================================================

print("\n--- 5. Few-Shot Final Classification ---")

few_shot_results, few_tokens, few_retries = classify_batch(
    eligible,
    few_shot=True,
    token_budget=FINAL_TOKEN_BUDGET,
    run_name="few_shot"
)

few_shot_results.to_csv(
    "outputs/triage_results.csv",
    index=False
)

print("Few-shot rows:", len(few_shot_results))
print("Few-shot token usage:", few_tokens)
print("Few-shot retries:", few_retries)
print(
    "Few-shot statuses:\n",
    few_shot_results["status"].value_counts().to_string()
)

status_counts = few_shot_results["status"].value_counts()

assert len(few_shot_results) == 164
assert status_counts.get("classified_json", 0) == 135
assert status_counts.get("classified_keyword", 0) == 12
assert status_counts.get("flagged_invalid_output", 0) == 9
assert status_counts.get("skipped_empty", 0) == 4
assert status_counts.get("skipped_long", 0) == 4

assert few_tokens <= FINAL_TOKEN_BUDGET

print("Final few-shot token budget:", FINAL_TOKEN_BUDGET)
print("Final few-shot actual token usage:", few_tokens)


# ============================================================
# TASK 7: ACCURACY AND CONFUSION MATRIX
# ============================================================

print("\n--- 6. Classification Evaluation ---")


def evaluate_classification(data, name):

    # The four known categories are the labelled classification task.
    evaluation_data = data[
        data["true_category"].isin(CATEGORIES)
    ].copy()

    actual = evaluation_data["true_category"].tolist()

    predicted = [
        value if value in CATEGORIES else "invalid"
        for value in evaluation_data["predicted_category"]
    ]

    labels = CATEGORIES + ["invalid"]

    accuracy = accuracy_score(actual, predicted)

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=labels
    )

    print(f"\n{name}")
    print("Evaluated tickets:", len(evaluation_data))
    print("Accuracy:", round(accuracy, 4))
    print("Confusion matrix labels:", labels)
    print(matrix)

    print(
        classification_report(
            actual,
            predicted,
            labels=labels,
            zero_division=0
        )
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    )

    matrix_df.to_csv(
        f"outputs/{name.lower().replace(' ', '_')}_confusion_matrix.csv"
    )

    return accuracy


zero_accuracy = evaluate_classification(
    zero_shot_results,
    "Zero Shot"
)

few_accuracy = evaluate_classification(
    few_shot_results,
    "Few Shot"
)

comparison = pd.DataFrame([
    {
        "approach": "Zero-shot",
        "accuracy": zero_accuracy,
        "token_usage": zero_tokens,
        "retry_count": zero_retries
    },
    {
        "approach": "Few-shot",
        "accuracy": few_accuracy,
        "token_usage": few_tokens,
        "retry_count": few_retries
    }
])

comparison.to_csv(
    "outputs/triage_fewshot_comparison.csv",
    index=False
)

print("\nZero-shot versus few-shot comparison:")
print(comparison.to_string(index=False))


# ============================================================
# TASK 8: TOKEN BUDGET EXHAUSTION DEMONSTRATION
# ============================================================

print("\n--- 7. Token Budget Exhaustion Demonstration ---")

budget_demo, budget_tokens, budget_retries = classify_batch(
    eligible,
    few_shot=True,
    token_budget=300,
    run_name="budget_demo"
)

budget_demo.to_csv(
    "outputs/triage_budget_demo.csv",
    index=False
)

budget_statuses = budget_demo["status"].value_counts()

print("Demo token budget: 300")
print("Demo token consumption:", budget_tokens)
print("Demo statuses:")
print(budget_statuses.to_string())

exhausted_count = int(
    (budget_demo["status"] == "budget_exhausted").sum()
)

print("Budget-exhausted tickets:", exhausted_count)

assert len(budget_demo) == 164
assert exhausted_count >= 1

print("Budget enforcement demonstration passed.")


# ============================================================
# TASK 9: HUMAN REVIEW AND FRAUD ROUTING
# ============================================================

print("\n--- 8. Human Review and Fraud Routing ---")

human_review = few_shot_results[
    few_shot_results["status"].isin([
        "flagged_invalid_output",
        "skipped_empty",
        "skipped_long",
        "budget_exhausted"
    ])
]

fraud_cases = few_shot_results[
    few_shot_results["predicted_category"] == "fraud"
]

print("Human review tickets:", len(human_review))
print("Fraud cases:", len(fraud_cases))

print("Fraud priority counts:")
print(fraud_cases["priority"].value_counts().to_string())

assert (
    fraud_cases["priority"] == "HIGH"
).all()

assert (
    fraud_cases["assigned_team"] == "Security Team"
).all()

human_review.to_csv(
    "outputs/triage_human_review.csv",
    index=False
)

fraud_cases.to_csv(
    "outputs/triage_fraud_security.csv",
    index=False
)


# ============================================================
# TASK 10: FINAL VALIDATION
# ============================================================

print("\n--- 9. Final Part C Validation ---")

assert len(few_shot_results) == 164

assert few_shot_results["ticket_id"].nunique() == 164

assert (
    few_shot_results["status"].isin([
        "classified_json",
        "classified_keyword",
        "flagged_invalid_output",
        "skipped_empty",
        "skipped_long",
        "budget_exhausted"
    ])
).all()

assert (
    few_shot_results["total_tokens"].sum() <= FINAL_TOKEN_BUDGET
)

print("All 164 ticket rows preserved.")
print("PII substitutions verified: 278.")
print("Empty tickets skipped: 4.")
print("Long tickets skipped: 4.")
print("Final few-shot token budget respected.")
print("Fraud routing verified.")
print("Retry log: outputs/triage_retry.log")

print("\n" + "=" * 65)
print("PART C PIPELINE COMPLETED")
print("=" * 65)
