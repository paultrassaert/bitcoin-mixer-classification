# Evaluation and implementation audit

Cell indices refer to the uploaded notebook, counting from zero and including markdown.

- Cells 39–41: baseline five-fold stratified CV and then a random 80/20 split within `final_train`. The reported baseline macro-F1 is 0.8903 ± 0.0041.
- Cell 44: the final XGBoost model uses `X_val` for early stopping. Cells 46 and 48 report F1/AUC on this same set. It has 6,537 rows and is not the separately supplied `test.csv`.
- `final_test` is extracted and checked, but there is no final model evaluation on it. `final_valid` is an unlabelled submission set; predicting it produces no accuracy estimate.
- Saved final class F1 values are approximately 0.95 (Bitcoin Fog), 0.96 (JoinMarket), 0.99 (Samourai), 0.87 (Wasabi), 0.99 (`other`). Strong aggregate results can hide weaknesses in smaller classes.
- The data health check tests duplicates inside each file, not between files. Shared transactions, entities or closely related graph components across splits need explicit investigation; the latter cannot be resolved without provenance.
- Cell 16: `fee_per_input` divides by the raw input count; the source uses fabricated `[0]` arrays when underlying values are missing. The refactor rejects incomplete records instead of silently interpreting missing amounts as zero.
- Cell 16: absence of a non-common-denomination output yields `-1 / n_outputs`, not a fixed sentinel. Revised code uses −1 consistently, adds an absence indicator and does not call the heuristic a verified change output.
- `max_repeated_output_value` was a frequency count despite its name; revised name is `max_equal_output_count`.
- The feature refactor is a portable replacement for the provider-specific extraction boundary, not a performance-preserving transformation. The old model must be retrained and tested on the new schema.

The optional XGBoost path preserves the model family while adding a separate provided-test evaluation. No actual labelled feature CSV was available during packaging. Only local integrity tests and a synthetic end-to-end Random Forest smoke test were run; synthetic metrics are not project results.

Private assignment boilerplate, API documentation, connector setup and any dataset records are excluded from the portfolio repository. The archived modelling notebook retains student analysis for provenance, with a prominent evaluation caveat and no saved outputs.
