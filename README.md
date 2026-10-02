# Bitcoin mixer-service classification

Five-class classification of Bitcoin transactions using input/output topology, output-value distributions, script types and fee characteristics. The original study used asynchronous feature acquisition, class-weighted XGBoost and SHAP.

**Author:** Paul Trassaert. This portfolio version contains provider-independent feature extraction and a stricter train/development/test evaluation workflow. The extraction refactor and audit were prepared with AI assistance. Private API connectors, interview instructions, reference documentation and datasets are not distributed.

## Task and historical results

Classes: `Bitcoin Fog`, `JoinMarket`, `SamouraiWallet.com`, `WasabiWallet.io`, `other`. These are service-attribution labels; they do not establish illegality or fraud.

The original feature extraction covered 32,682 training, 16,176 labelled test and 16,392 unlabelled validation transactions. The training distribution was imbalanced (approximately 85.2% `other`). Saved baseline five-fold stratified CV macro-F1 was **0.8903 ± 0.0041**.

The later notebook reported **macro-F1 approximately 0.95** and mean one-vs-rest ROC-AUC **0.9981** on 6,537 examples drawn from the training file. **That same internal holdout was used for early stopping**, and the notebook describes iterative experimentation. These are model-selection diagnostics, not an untouched test result. Although `final_test` was extracted, it was never evaluated by the supplied notebook. The raw feature files are unavailable in this repository, so those reported scores have not been independently reproduced.

## Research walkthrough: choices, figures and interpretation

This walkthrough brings the author's reasoning directly beside the original notebook figures: what was tried, what was observed, and why the next modelling choice followed. Wording has been edited for clarity and evaluation claims have been corrected against the source. [The original wording](docs/original-analysis.md) remains available in full. These are saved historical outputs, not results from a new training run.

### 1. Building the transaction features

The original pipeline acquired transaction information asynchronously and processed requests in **batches of 2,000 transactions**. The motivation was practical: avoid loading an entire remote extraction at once and reduce the risk of memory pressure or timeouts. The notebook then computed features describing input/output counts, repeated output denominations, entropy, script types, transaction size and locktime.

The original health check reported no missing or infinite values and included edge cases such as zero-output transactions. These checks establish numerical consistency for that extraction; they do not by themselves validate the service labels or establish that train and test transactions are independent. Intermediate features were kept in memory for the assignment. Persisting versioned feature tables would make later experiments easier to reproduce without querying the provider again.

### 2. Class imbalance: why accuracy was insufficient

![Class distribution](assets/original/cell-25-figure-1.png)

The first important observation was that **85.2% of the training examples belong to `other`, while JoinMarket accounts for only about 1.3%**. A classifier could therefore obtain roughly 85% accuracy by ignoring every minority service. That would not solve the attribution problem.

This motivated macro-F1, per-class recall and one-vs-rest ROC curves, rather than headline accuracy alone. Macro averaging gives each class equal weight; weighted or micro averages do not. The plot's original title, “Market Share”, refers only to this dataset's composition.

The author deliberately avoided SMOTE: interpolated numerical features might not correspond to structurally valid Bitcoin transactions. Class/sample weights were used instead to penalise mistakes on minority labels more strongly. This is a modelling choice to test, not a guarantee that minority-class detection will improve.

### 3. Distributions: looking for transaction structure

![Feature distributions](assets/original/cell-28-figure-1.png)

The histograms suggested that **input/output counts and output-denomination structure carry useful service information**. In the author's analysis, Samourai and Wasabi examples often had more inputs and outputs than the `other` examples, while repeated denominations changed the distributions of `unique_output_ratio` and output-value entropy. The `p2wpkh_ratio` also appeared useful for separating particular labelled services.

The reasoning was to describe the construction of a transaction rather than simply its monetary size. Similar-denomination outputs and script choices can leave statistical patterns that a tabular model can exploit. These are patterns in this sample, not universal rules defining a service.

![Entropy versus output count](assets/original/cell-31-figure-1.png)

The joint plot tests whether output counts and entropy provide a clearer picture together than separately. The author's reading was that the `other` population was more dispersed, whereas some Wasabi and JoinMarket observations formed tighter groups. This motivated keeping complementary structural features instead of expecting one entropy threshold to classify every transaction.

Here, entropy is calculated from **frequencies of output denominations**, not from each output's share of the total Bitcoin amount. That definition matters when interpreting repeated amounts and comparing this plot with other implementations.

### 4. Correlations and feature pruning

![Feature correlations](assets/original/cell-34-figure-1.png)

The correlation matrix highlighted expected redundancy: for example, transactions with more inputs tend to occupy more virtual bytes. The author used this inspection to remove highly redundant features and experimental feature combinations that appeared to add noise rather than useful discrimination.

The original interpretation described the remaining features as independent. More precisely, low pairwise linear correlation only suggests that they are not simple linear duplicates; it does not prove statistical independence or incremental predictive value. The appropriate next check is whether removing a feature changes validation performance consistently.

### 5. Why XGBoost, and what else was attempted

The author focused on XGBoost because the task is tabular, nonlinear interactions are plausible, and SHAP can help inspect the fitted model. Stratified splits retained minority classes, and sample weights addressed the imbalance. The saved baseline five-fold macro-F1 was **0.8903 ± 0.0041**. Similar fold scores are reassuring about split-to-split variability, but do not prove that feature extraction or dataset separation is correct.

Hyperparameters were explored in several stages, narrowing subsequent searches around promising candidates. Early stopping limited additional trees when the internal development score stopped improving. The original analysis also records attempts with **Random Forest, RF/XGBoost voting and stacking**; these did not improve the author's observed results. Stacking introduced an additional model whose tuning was difficult. Increasing the number of trees while lowering the learning rate also failed to produce a monotonic improvement: a smaller learning rate alone was not a solution.

These observations explain why the submitted notebook retained a single XGBoost model. They are reported experimental observations, not a controlled benchmark establishing that XGBoost always outperforms those alternatives.

### 6. Confusion matrix: the remaining Wasabi errors

![Normalised confusion matrix](assets/original/cell-47-figure-1.png)

The author identified **Wasabi versus `other`** as the main remaining confusion: roughly one in ten Wasabi examples was assigned to `other` in the saved diagnostic. This is more informative than a single aggregate score because it identifies which service and which type of error need attention.

The author's hypothesis was that further progress would require better features grounded in Bitcoin/CoinJoin mechanics, rather than simply more hyperparameter trials. Additional Wasabi-oriented heuristics were attempted but degraded the observed score, so they were not retained. A useful next experiment would inspect the false negatives, compare their structural features with correctly classified Wasabi transactions, and test a small number of documented hypotheses on development data.

![One-vs-rest ROC curves](assets/original/cell-48-figure-1.png)

The notebook reported macro-F1 around **0.95** and mean one-vs-rest AUC **0.9981** on 6,537 examples. These plots show strong separation **within that development sample**. The same sample was used for early stopping and the notebook describes repeated experimentation, so those values must not be presented as independent final-test performance. The separate labelled `final_test` was extracted but never scored by the supplied notebook.

### 7. SHAP: what surprised the author

![Global SHAP importance](assets/original/cell-52-figure-1.png)

The most interesting discrepancy between intuition and the fitted model concerned **entropy**. The author initially expected entropy to be a dominant discriminator, yet the SHAP analysis made it look less influential than anticipated. In contrast, **`n_locktime`** was particularly influential in the author's interpretation, especially for Bitcoin Fog and JoinMarket.

This suggested that implementation-related transaction fields could contain stronger sample-specific signatures than an intuitive “mixing complexity” measure. However, a SHAP ranking measures how the fitted model uses the available features. Correlated alternatives can share or substitute for information, so a lower ranking does not establish that entropy has no value. Nor does the importance of locktime prove that a service uses old software; that was an original hypothesis requiring separate evidence.

### 8. Class-specific explanations

Each plot below asks which features push the model toward or away from **that particular class**. A positive SHAP contribution supports that class output relative to the explainer's baseline; it is not a universal “mixer” score. Point colour represents the feature value, while horizontal position represents its contribution.

![SHAP: Bitcoin Fog](assets/original/cell-54-figure-1.png)

For **Bitcoin Fog**, the author particularly highlighted locktime. Reading colour and contribution together helps determine which observed values support or oppose this class. The interpretation is a learned attribution pattern, not evidence about the service's code history.

![SHAP: JoinMarket](assets/original/cell-54-figure-2.png)

For **JoinMarket**, locktime was also a notable observation. The class-specific view matters because the same field need not have the same direction or strength for Bitcoin Fog and JoinMarket. With JoinMarket rare in the dataset, this explanation should be considered alongside its support and per-class recall.

![SHAP: SamouraiWallet.com](assets/original/cell-54-figure-3.png)

For **Samourai**, the purpose is to relate class contributions back to the script and input/output patterns found during exploration. Features can be useful jointly even if no single threshold separates every service. The plot explains the fitted model's decisions; it does not establish a deterministic transaction rule.

![SHAP: WasabiWallet.io](assets/original/cell-54-figure-4.png)

For **Wasabi**, this plot connects explainability to the confusion matrix: the important follow-up is to compare explanations for correctly identified transactions and false negatives assigned to `other`. The supplied plot summarises contributions for this class; it does not itself perform that error-subgroup comparison.

![SHAP: other](assets/original/cell-54-figure-5.png)

For **`other`**, the plot shows what makes a transaction resemble the majority label in this training setup. `other` is a heterogeneous residual class, not a single protocol or service. A strong contribution toward it should therefore be interpreted as a classification decision, not a claim that a transaction is ordinary, legitimate or unrelated to privacy tools.

### 9. What the project established, and the next evaluation

The project demonstrates a complete path from transaction feature engineering to class-aware evaluation and explainability. Its most substantive findings are the imbalance problem, the residual Wasabi errors, and the gap between intuitive entropy-based features and the model's use of locktime and other structural fields.

To establish performance beyond the development sample, freeze the feature schema and model choices, check transaction/entity or temporal overlap as appropriate, and evaluate the separate labelled test table once. Report per-class support, precision, recall and macro-F1 alongside the confusion matrix. Keep the unlabelled validation predictions separate from claims of measured accuracy. The original notebook ends with validation-set prediction; predictions without labels cannot supply an additional performance score.


## Run

Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
python train.py --train-features data/train_features.csv --test-features data/test_features.csv --out results/run
```

Provide your authorised, locally exported feature tables. Both files require unique `txid`, one of the five labels in `label`, and matching numeric feature columns. Metadata and redundant columns listed in `EXCLUDED` are removed explicitly. The command rejects cross-split transaction overlap, schema mismatches and non-finite features. Exact transaction separation does **not** establish graph/entity/temporal independence; those checks require additional provenance.

The default is a fixed Random Forest reference. For XGBoost, install the optional dependency and select it:

```bash
python -m pip install -r requirements-xgboost.txt
python train.py --model xgboost --train-features data/train_features.csv --test-features data/test_features.csv --out results/xgb_run
```

The script splits the training file into fit/development sets. Only development labels can affect early stopping. The provided test file is evaluated after fitting. Do not tune the model after inspecting the reported test metrics and continue calling that file untouched. The random internal split is a convenience baseline; future entity-aware or chronological validation should follow the actual deployment question.

## Feature extraction

`features.transaction_features(record)` accepts a local dictionary with:

```python
record = {
    'txid': 'example',
    'input_values_sats': [401000],
    'output_values_sats': [100000, 100000, 100000, 100000],
    'input_scripts': ['P2WPKH'],
    'output_scripts': ['P2WPKH'] * 4,
    'vsize': 200,
    'locktime': 0,
}
```

All amounts use integer satoshis. Non-coinbase inputs, positive size, matching script/value lengths and non-negative fees are validated. Entropy describes frequencies of output amounts, not monetary shares.

The old `change_pos_relative` mixed a missing sentinel with the number of outputs. The revised proxy uses a consistent −1 for absence and a separate indicator. It is named `first_nondenom_relative_position`: a non-standard denomination is **not proof of a change output**. `max_equal_output_count` explicitly denotes a count, not a Bitcoin amount. Because some feature definitions changed, old trained models must not be loaded against this schema without retraining.

## Repository contents

- `features.py`: local, provider-independent feature extraction.
- `train.py`: separate development/test evaluation; Random Forest and optional XGBoost.
- `test_features.py`: amount, sentinel and cross-split integrity checks.
- `notebooks/portfolio_walkthrough.ipynb`: feature demonstration and evaluation instructions.
- `archive/original_model_analysis.ipynb`: original student modelling cells with outputs stripped, supplied as historical context. Requires original exported feature tables and optional plotting/SHAP packages; it is not the supported reproduction path.
- `AUDIT.md`: provenance and evaluation limitations.

Real training/test data was not supplied, so only feature/split tests and a clearly synthetic Random Forest execution check could be run during this packaging. The XGBoost path was not executed in this environment. No new real-data performance score is claimed.

No licence is applied to third-party data or interview materials. No proprietary material is included in the supported implementation.
