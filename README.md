# Instamart Customer Intelligence

## Overview

This repository uses the deterministic synthetic exports from `generate_data.py` as its data source. The generated data does **not** match the numeric reference checkpoints in the assignment: for example, the generator writes 13,846 order rows and 13,084 distinct IDs. To keep the raw exports authoritative, this project reports the results produced from those exports rather than changing or fabricating them to fit the references. The actual audit and analysis results are recorded below.

## Reproduce

Run these commands from the repository root in Windows PowerShell. The supplied `generate_data.py` and `mock_llm.py` are at the repository root.

1. Check out the repository's `main` branch.

   ```powershell
   git switch main
   ```

2. Create and activate an isolated virtual environment, install the frozen libraries, and refresh the freeze if the environment changes.

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   python -m pip freeze > requirements.txt
   ```

3. Regenerate all deterministic raw exports and policy files.

   ```powershell
   python generate_data.py
   ```

4. Audit the raw exports, clean the customers, products and orders, save CSV and SQLite tables, and run the seven SQL queries. The pipeline also generates the support-ticket and policy-assistant outputs; Part B's segmentation and lapse workflow runs separately below.

   ```powershell
   python src\instamart_pipeline.py
   ```

5. Validate the cleaned CSVs and SQLite tables.

   ```powershell
   python src\validate_data.py
   ```

   On success the script prints exactly `ALL CHECKPOINTS PASSED`.

   The validator checks the cleaned-data invariants and CSV/SQLite consistency; it does not assert the assignment's incompatible fixed reference counts.

6. Re-run the SQL and pandas analysis report.

   ```powershell
   python src\run_analysis.py
   ```

The first clean run creates `data/clean/customers.csv`, `data/clean/products.csv`, `data/clean/orders.csv`, and `data/instamart.db`. The database contains `customers`, `products`, and `orders`; the `order_ts` values are stored as `YYYY-MM-DD HH:MM:SS` text. `data/raw/tickets.csv` is read with `keep_default_na=False`, preserving empty ticket text as an empty string.

The committed deliverables include the SQL queries in `sql/queries.sql`; Part B segments in `outputs/customer_segments.csv`; Part C results in `outputs/triage_results.csv` and `outputs/triage_budget_demo.csv`; and Part D scenario runs and refund actions in `outputs/agent_runs.jsonl` and `outputs/refund_ledger.jsonl`. The Part D crash/resume checkpoint is `state/crash_resume_S1.json`.

## Part A

### Raw audit

Pandas inferred the following raw shapes and data types. Missing percentages are calculated as missing-value count divided by row count; for the ticket table, empty strings are not converted to missing values.

| Table | Shape | Column | Inferred dtype | Missing count | Missing % |
|---|---:|---|---|---:|---:|
| customers | 1,200 × 5 | customer_id | object | 0 | 0.00% |
| customers | 1,200 × 5 | city | object | 0 | 0.00% |
| customers | 1,200 × 5 | gender | object | 14 | 1.17% |
| customers | 1,200 × 5 | age | object | 57 | 4.75% |
| customers | 1,200 × 5 | signup_date | object | 0 | 0.00% |
| products | 40 × 4 | product_id | object | 0 | 0.00% |
| products | 40 × 4 | product_name | object | 0 | 0.00% |
| products | 40 × 4 | category | object | 0 | 0.00% |
| products | 40 × 4 | price_inr | int64 | 0 | 0.00% |
| orders | 13,846 × 9 | order_id | object | 0 | 0.00% |
| orders | 13,846 × 9 | customer_id | object | 0 | 0.00% |
| orders | 13,846 × 9 | product_id | object | 0 | 0.00% |
| orders | 13,846 × 9 | quantity | int64 | 0 | 0.00% |
| orders | 13,846 × 9 | order_ts | object | 0 | 0.00% |
| orders | 13,846 × 9 | status | object | 0 | 0.00% |
| orders | 13,846 × 9 | discount_pct | int64 | 0 | 0.00% |
| orders | 13,846 × 9 | order_amount | object | 60 | 0.43% |
| orders | 13,846 × 9 | delivery_minutes | float64 | 2,547 | 18.40% |

The generator's raw figures, calculated from the generated exports, are:

| Audit measure | Actual value |
|---|---:|
| Raw order rows | 13,846 |
| Exact duplicate rows beyond the first copy | 114 |
| Rows with status `pending` | 2,160 |
| Distinct `order_id` values | 13,084 |
| `order_amount` values that `to_numeric(errors="coerce")` cannot convert (including blanks) | 105 |
| Negative `delivery_minutes` | 91 |
| Blank/missing `delivery_minutes` | 2,547 |
| Blank customer ages | 57 |
| Numeric ages outside 10–100 | 10 |
| Distinct raw city spellings | 16 |
| Distinct nonblank raw gender spellings | 9 |
| Blank genders | 14 |

These are not the supplied reference values (13,084 rows, 119 duplicates, 656 pending rows, 12,309 IDs, and the other listed reference figures). The generator iterates over 13,084 distinct base order IDs and then adds duplicate and pending records, which is why its raw row and status counts differ.

### Cleaning rules and measured changes

#### Customers

| Rule | Changes on this generated export |
|---|---|
| Strip leading/trailing whitespace from `city` and apply title case. | 32 city values changed; the normalized data has six cities. |
| Normalize gender spellings (`male`/`m`, `female`/`f`) to `male` or `female`; map blanks and other spellings to `unknown`. | 28 values changed, including the 14 blanks. |
| Convert age values to numeric; treat nonnumeric and numeric values outside 10–100 as missing; fill missing values with each city’s median using `groupby("city")["age"].transform("median")`. | 68 ages were missing after conversion/range checking (57 blank, 10 numeric out of range, 1 nonnumeric token); all 68 were filled. The final age range is 18–70. |
| Convert `signup_date` to datetime. | 1,200 values converted; 0 failed conversions. |

Clean customer counts from this generator are:

| City | Customers |
|---|---:|
| Hyderabad | 221 |
| Pune | 208 |
| Mumbai | 200 |
| Delhi | 194 |
| Chennai | 193 |
| Bengaluru | 184 |

| Gender | Customers |
|---|---:|
| female | 600 |
| male | 586 |
| unknown | 14 |

#### Products

Strip whitespace from `product_name` and `category` (0 values changed in this export), and convert all 40 `price_inr` values to numeric (0 conversion failures).

#### Orders

Rules are applied in this order:

1. Convert all 13,846 `order_ts` values to datetime; 0 conversions failed.
2. Remove exact duplicate rows, keeping the first copy; 114 rows removed.
3. Sort by `order_id` and `order_ts`, then keep the latest row for each ID; 648 more duplicate-ID rows removed. The generator leaves 1,501 IDs whose latest status is still `pending`; discard those because the required clean order table has no pending status. This removes 1,501 additional rows.
4. Convert `order_amount` to numeric, then fill 96 missing/unparseable amounts after deduplication and pending-row removal with `round(price_inr × quantity × (1 − discount_pct / 100), 2)`.
5. Set the 89 negative delivery times remaining after row selection to missing. Among the retained completed orders, fill all 737 missing delivery times with the median for their product category, computed from completed orders using `groupby` and `transform`. Keep all 1,149 cancelled-order delivery times missing; do not invent a delivery time for an undelivered order.

The resulting clean orders table has 11,583 rows: 10,434 completed and 1,149 cancelled, no pending rows, and no missing order amounts. There are no missing delivery times among completed orders and 1,149 among cancelled orders. Completed-order revenue sums to **INR 5,240,290.80** for this generated data; this differs from the supplied reference total of INR 2,998,668.80.

### SQL analysis

The seven runnable statements are saved in [`sql/queries.sql`](./sql/queries.sql), and `src/run_analysis.py` executes each with `pandas.read_sql_query` against `data/instamart.db`. Results on the generated exports:

| Query | Actual result |
|---|---|
| Q1 — completed orders over INR 500, grouped by customer with totals over INR 8,000 | 83 customers |
| Q2 — completed revenue by city and category | 30 rows; total INR 5,240,290.80; largest cell Hyderabad / Electronics, INR 669,047.70 |
| Q3 — products priced above average catalogue price | 11 products |
| Q4 — customers with no completed order | 0 customers |
| Q5 — customers above average completed spend | 478 customers; average customer completed spend INR 4,366.91 |
| Q6 — completed spend and ranking windows by city | 1,200 customer rows; 1 row has `RANK()` different from `ROW_NUMBER()`. A tie shares a `RANK()` position, so subsequent ranks can skip while `ROW_NUMBER()` remains unique. |
| Q7 — daily completed revenue and running total | 451 daily rows; last cumulative value INR 5,240,290.80 |

Q6 top spender in each city:

| City | Customer | Completed spend |
|---|---|---:|
| Bengaluru | C0102 | INR 13,328.50 |
| Chennai | C0026 | INR 25,455.55 |
| Delhi | C0197 | INR 18,598.85 |
| Hyderabad | C0175 | INR 18,939.15 |
| Mumbai | C0725 | INR 15,742.25 |
| Pune | C0078 | INR 22,596.00 |

### pandas cross-check and exploratory analysis

`run_analysis.py` reconstructs Q2 using `pd.pivot_table(..., margins=True)`. The pivot grand total is INR 5,240,290.80, matching SQL. Per-city completed-order `groupby` named aggregation:

| City | Sum | Mean | Count |
|---|---:|---:|---:|
| Bengaluru | INR 808,595.70 | INR 503.80 | 1,605 |
| Chennai | INR 833,936.25 | INR 479.55 | 1,739 |
| Delhi | INR 878,367.30 | INR 545.23 | 1,611 |
| Hyderabad | INR 1,021,009.45 | INR 522.26 | 1,955 |
| Mumbai | INR 829,819.40 | INR 472.29 | 1,757 |
| Pune | INR 868,562.70 | INR 491.55 | 1,767 |

For completed `order_amount`: mean INR 502.23; median INR 201.60; skewness 4.7753; Q1 INR 85.50; Q2 INR 201.60; Q3 INR 490.00; IQR INR 404.50; lower and upper 1.5×IQR fences INR -521.25 and INR 1,096.75; 0 outliers below the lower fence and 1,195 above the upper fence.

The data describes customer grocery and retail orders, customers, products, discounts, and delivery times. Completed order amounts are strongly right-skewed: the mean (INR 502.23) is well above the median (INR 201.60), and 1,195 amounts exceed the upper outlier fence. The largest absolute correlation is between `order_amount` and `quantity` (+0.3370); the amount-discount correlation is -0.0794. Correlation of this size does not establish causation.

The actual results above also differ from the assignment references, including Q1 (83 vs. 30), Q4 (0 vs. 25), Q5 (478 vs. 550), Q7 (451 vs. 183), the expected city-level spend values, the expected distribution statistics, and the expected largest-correlation pair. These divergences originate in the generated raw exports and are intentionally not hidden by modifying the source data.

## Part B — RFM segments and lapse assessment

Run Part B after Part A has generated the clean CSVs:

```powershell
python src\run_part_b.py
```

This computes whole-window RFM as of 2025-10-01, records K-means inertia and silhouette for K=2–8, profiles three reproducible segments, runs PCA on the scaled RFM features, and writes `outputs/customer_segments.csv` for downstream ticket routing. K=3 is selected because this data's inertia drop bends after K=3 and K=3 also has the highest silhouette score; three campaign groups are straightforward to action. The Premium flag is true only for the cluster with the highest mean monetary value. Cluster IDs are arbitrary labels and do not imply rank or value. The Premium flag is a post-hoc whole-window label that includes the September lapse window, so any retention list filtered by it is illustrative and not a leakage-free evaluation. The current data’s cluster metrics and profiles print when the command runs.

The actual results from the Part A exports are:

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 2,375.9 | 0.310 |
| 3 | 1,719.4 | 0.325 |
| 4 | 1,407.3 | 0.295 |
| 5 | 1,233.5 | 0.294 |
| 6 | 1,077.4 | 0.270 |
| 7 | 952.6 | 0.281 |
| 8 | 870.3 | 0.278 |

With this dataset, inertia’s largest decrease is from K=2 to K=3, then its improvements taper, and K=3 also has the highest silhouette among the tested choices. K=8 has lower inertia than K=2 because adding centroids can only reduce or preserve within-cluster squared distances; inertia has no penalty for extra clusters. Inertia alone therefore always favors more clusters and does not tell the business whether the additional groups are stable or actionable.

RFM includes 1,200 customers; recency median/min/max is 588/553/955 days, frequency mean/max is 8.70/19, and monetary mean/max is INR 4,366.91/INR 25,455.55. This differs from the task references because the latest completed order is in 2024, not near the 2025 snapshot.

| Segment ID | Name | Size | Mean recency (days) | Mean frequency | Mean monetary | Recommended action |
|---:|---|---:|---:|---:|---:|---|
| 0 | Premium | 350 | 585.29 | 11.63 | INR 7,666.52 | Protect loyalty with VIP service and exclusive early access. |
| 1 | Win-Back Candidates | 234 | 688.25 | 6.37 | INR 2,707.43 | Use a personalized win-back message and limited-time return incentive. |
| 2 | Frequent Loyalists | 616 | 584.04 | 7.91 | INR 3,122.51 | Reward repeat orders with replenishment reminders and a multi-order loyalty offer. |

PCA on the scaled RFM matrix explains 0.560 and 0.276 of variance in the first two components (total 0.836). The lapse cohort has 1,200 customers, all 1,200 labeled lapsed (100%); there are no completed orders in the September label window. The always-not-lapsed baseline therefore has 0.000 accuracy and 0.000 lapsed-class F1 for these actual labels. The required feature check finds 0 missing values, but stratification is impossible with one target class.

For lapse assessment, the cutoff is 2025-09-01. The cohort and model features use completed and final-status orders strictly before the cutoff; September is considered only when making the target label. The task 3 segment labels are not included in the lapse features. A feature missing-value check runs before model training. For lapse preprocessing, no encoder, scaler or resampler saw test rows; however, the supplied data has no completed orders in September 2025, so every cohort customer is labeled lapsed and a test split cannot be made. Therefore the requested grid searches, model comparisons, oversampling experiment, feature interpretation, persisted lapse model and retention list cannot be validly produced from this single-class target. The script reports the baseline and stops before making those artifacts rather than inventing labels or model results. If source-authoritative data with both classes is supplied, the script is wired to run the specified stratified split, training-only preprocessing, five-fold searches (45 random-forest fits), training-only oversampling experiment, evaluation, model persistence and retention-list generation.

## Part C — Privacy-preserving ticket triage

Run the triage pipeline from the repository root:

```powershell
python -m src.triage_tickets
```

The pipeline reads `data/raw/tickets.csv` in file order, rejects whitespace-only and over-400-token texts before a model call, and checks the user message is exactly the redacted ticket text. It applies the requested email, phone, and six-digit order-ID expressions in order and asserts that none remain. The supplied ticket text contains 131 additional `ORD-` IDs with only 3–5 digits, which the required `ORD-\d{6}` expression does not match; a conservative fallback redacts those as `[ORDER_ID]` too, and an assertion rejects any remaining `ORD-` followed by digits. For this supplied file the measured counts are 65 emails, 73 phones, 0 six-digit IDs, and 131 fallback IDs: 269 substitutions total, with zero matching personal-data patterns left in redacted text. The three specified expressions alone would match 138 substitutions, not the assignment's cited 278.

The zero-shot and few-shot system prompts are defined in `src/triage_tickets.py`. Each model call retries transient failures at most three times with the specified exponential jitter; one retry line includes ticket ID, retry number, and wait. Outputs are accepted only as JSON with an allowed category or, for non-JSON replies only, when exactly one allowed category word is found. In a full run, 19 transient failures were recovered; 135 answers were JSON-validated, 12 were keyword-validated, and 9 were sent to human review as invalid output. The four empty and four long tickets were also not sent to the model.

Measured on the checked-in tickets, zero-shot classified 147 tickets at 81.0% accuracy (119/147) using 9,584 total response tokens, or 65.20 tokens per classified ticket. Its confusion table is:

| True \ Predicted | billing | delivery | fraud | technical |
|---|---:|---:|---:|---:|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 38 | 0 | 0 |
| fraud | 12 | 0 | 23 | 0 |
| technical | 16 | 0 | 0 | 20 |

Few-shot classified 147 at 100.0% accuracy (147/147) using 18,320 total response tokens, or 124.63 per classified ticket. It costs 8,736 extra tokens overall (59.43 more per classified ticket, about 91% more than zero-shot), while raising measured accuracy by 19.0 percentage points; that tradeoff is worthwhile when avoiding these classification errors matters more than the added token cost. These measured email/phone/order counts and zero-shot accuracy differ from the assignment references because the supplied dataset has different text and labels; the reported values are computed from the checked-in file rather than adjusted to match the references.

The zero-shot budget demo uses 6,000 tokens and marks 58 tickets `budget_exhausted`, writing `outputs/triage_budget_demo.csv`. The final few-shot run uses a 100,000-token budget and writes the 164-row `outputs/triage_results.csv`. JSON- and keyword-validated categories are rule-validated model outputs, not human approvals. Empty, long, invalid-output, and budget-exhausted tickets route to `human_review` with normal priority; fraud routes to `security_team`, premium customers otherwise route to `premium_desk`, and remaining classified tickets route to `support_queue`.

## Part D — Policy assistant and refund agent

Run the retrieval evaluation, seven scenarios, safe-stop demonstrations, and crash/resume demonstration with:

```powershell
python -m src.policy_agent
```

Documents are split into 40-word chunks with 10-word overlap: 15 total (refund 4, delivery 3, escalation 3, payment 3, privacy 2). TF measures how often a term occurs in a chunk, while IDF gives less weight to terms common across many chunks and more weight to rarer terms. TF-IDF is lexical, so it may rank poorly when a question and a relevant policy sentence use different words for the same idea.

| Chunk overlap | Answerable source in top 3 | Expected fact in answer | Unanswerable fallback |
|---|---:|---:|---:|
| 10 words | 12/12 | 12/12 | 2/2 |
| 0 words | 12/12 | 10/12 | 2/2 |

With zero overlap, “What delivery time makes a completed order eligible for a full refund?” loses the answer phrase `more than 45 minutes`: the 40-word boundary splits the sentence after “the customer,” so the mock returns only “receives a full refund of the order amount.” The payment-failure answer also splits at `within 48 hours`. The policy assistant passes only the top three retrieved chunks in its specified `CONTEXT:`/`QUESTION:` user message and returns the mock's answer and source list.

The seven scenario outcomes use the actual checked-in scenario file and database:

| Scenario | Status | Refund amount (INR) | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | 569.05 | 5 | 1,042 |
| S2 | REFUND | 14.40 | 5 | 1,042 |
| S3 | ESCALATE | 1,039.20 | 5 | 1,029 |
| S4 | DENIED | 0.00 | 4 | 721 |
| S5 | DENIED | 0.00 | 3 | 424 |
| S6 | NOT_FOUND | 0.00 | 2 | 130 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,174 |

The supplied data's S1–S3 expected amounts are INR 569.05, 14.40, and 1,039.20, respectively; these agree with the database and `agent_scenarios.csv` but differ from the alternate reference amounts in the task description. Because the unmodified mock parses six-digit order IDs only, the agent formats shorter scenario IDs as six-digit lookup aliases before sending them to the mock; `get_order` resolves the alias to the actual database ID before recording an action. The tool functions independently validate that automatic refunds are for completed, late-delivered orders, match the full order amount, and do not exceed INR 1000; larger eligible amounts are recorded only through human escalation. All simulated requests are treated as being inside the seven-day policy window.

| Safety demonstration | Result |
|---|---|
| S1 with `token_limit=200` | `FALLBACK_TOKEN_LIMIT` |
| S1 with `time_limit_seconds=0` | `FALLBACK_TIMEOUT` |
| S7 with default six-iteration limit and 6,000-token budget | `FALLBACK_MAX_ITERATIONS` after 6 calls |

The simulated S1 crash occurs after `get_order` and `search_policy` have both been executed and checkpointed; resume continues with `calculate_refund` and `issue_refund`. The complete ordered tool list is `get_order` → `search_policy` → `calculate_refund` → `issue_refund`; there are four executions total and none is repeated. Checkpoints are written under `state/`; the seven scenario runs are in `outputs/agent_runs.jsonl`, and only validated issued/escalated actions are appended to `outputs/refund_ledger.jsonl`.

**ReAct versus plan-and-execute (under 250 words).** The measured ReAct flow used 5 model calls/1,042 tokens for S1, 3/424 for cancelled-order S5, and 2/130 for missing-order S6. A plan-and-execute design could ask once for a complete plan, then perform its known tools without another model call, using fewer calls and usually fewer tokens on straightforward requests; it would need a new model call only when a step fails or the observation differs from the plan. However, an unexpected `NOT_FOUND` result or a cancelled-order observation can invalidate downstream steps: a fixed plan risks attempting a calculation or refund that no longer applies unless the executor independently guards every action. ReAct observes each tool result before requesting the next action, and the refund tools also enforce completion, delivery-time, and amount rules. I would choose ReAct for refunds because the additional calls buy explicit observation-driven decisions and safer handling of missing, cancelled, or otherwise unexpected orders; the high-impact refund action should not be based solely on a precomputed plan.

## Limitations and Next Steps

The source generator creates orders only from 2023-01-01 through 2024-03-26. Thus it cannot support the September 2025 lapse outcome window or the Part A/Part B reference figures tied to later data. Replace the supplied clean exports with source-authoritative data spanning the specified windows and containing both lapse classes, then rerun Part A and Part B to perform leakage-safe model tuning and evaluation. The Premium flag is a post-hoc whole-window label that includes September orders; any retention list filtered by that flag is illustrative and is not a leakage-free evaluation.
