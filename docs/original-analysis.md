# Original analysis and figures

These are the author/team-written analysis cells and saved figures from the supplied original notebook, preserved in their original order. Source cell numbers are zero-based. The figures are historical saved outputs, not newly rerun experiments. Course/interview instructions, private API documentation, data and connection code are excluded.

**Reading notes:** service classification does not establish fraud or illegality. “Market Share” in the original plot describes this dataset only. The internal holdout was also used for early stopping, so its scores are model-selection diagnostics. Correlation and SHAP values do not establish independence, causality, or a service's implementation history. See [the audit](../AUDIT.md) for the current interpretation.

<!-- Original notebook cell 12 -->

## 1. Feature Extraction

<!-- Original notebook cell 13 -->

### 1.1 Importation

<!-- Original notebook cell 15 -->

### 1.2 Creation of features

<!-- Original notebook cell 17 -->

### 1.3 Extraction

<!-- Original notebook cell 18 -->

For data extraction, I implemented a Batch Processing strategy with chunks. Since I am sending requests to a remote database, to avoid saturating the RAM or causing network timeouts, I split the submissions into blocks of 2000 transactions. This method is intended to guarantee the stability of the pipeline.

<!-- Original notebook cell 20 -->

### 1.4 Verification

<!-- Original notebook cell 22 -->

The health check is primarily there to validate the code and ensure that the manipulations performed on the data (calculate the entropy, ratios...) did not negatively alter it through side effects or unintended phenomena. It confirms a perfectly clean dataset, with zero missing values (NaNs) and no infinite values (Infs).

Moreover, since the features were extracted from the Bitcoin blockchain using real transactions IDs, the dataset benefits from the structural consistency of the ledger itself. This guarantee a high level of reliability and confidence. Furthermore, the extraction engine successfully handled edge cases, such as zero-output transactions, during the feature engineering phase. As a result, no additional cleaning or imputation is required.

For this assignment, all features are processed and stored in memory to keep the notebook simple and smooth.
In a production setup, I would usually save intermediate files (e.g., .csv) for safety. However, since the extraction process is fast, we can easily recompute everything if needed.

<!-- Original notebook cell 23 -->

## 2. Exploratory Data Analysis

<!-- Original notebook cell 24 -->

### 2.1 Distribution

![Class distribution](../assets/original/cell-25-figure-1.png)

*Class distribution. Original saved output, cell 25.*

<!-- Original notebook cell 26 -->

It can be noted that the dataset is heavily imbalanced; the 'other' label is represented at 85.2%, while the rest are mixers and, in particular, JoinMarket accounts for only 1.3% of the transactions. 

This structure poses a major risk: a naive model could achieve 85% accuracy simply by systematically predicting the majority class, thus failing entirely to detect fraudsters, which is nonetheless our primary objective. 

To obtain the best scores, we must therefore focus as much as possible on the F1-score and the ROC-AUC, which treat each label with the same importance. 
Furthermore, I decided not to use SMOTE due to the complexity of the data and also because these are real transactions, and I believe they are extremely difficult, if not impossible, to reproduce while maintaining exact labels.

<!-- Original notebook cell 27 -->

### 2.2 Histograms

![Feature distributions](../assets/original/cell-28-figure-1.png)

*Feature distributions. Original saved output, cell 28.*

<!-- Original notebook cell 29 -->

The feature distributions reveal high separability between regular transactions and mixer services.
Mixers (especially Samourai and Wasabi) involve more inputs and outputs than the "other" class, which is concentrated at minimal values.
The unique_output_ratio and entropy distributions show a clear collapse for mixers, that confirms the use of standardized output amounts, a key heuristic for detection.
The p2wpkh_ratio acts as a strong categorial discriminator. It means that transactions have specific patterns that make them easy to recognize as coming from Samourai or JoinMarket.

<!-- Original notebook cell 30 -->

### 2.3 Behavioral Clutering

![Entropy versus output count](../assets/original/cell-31-figure-1.png)

*Entropy versus output count. Original saved output, cell 31.*

<!-- Original notebook cell 32 -->

The analysis of entropy versus number of outputs shows characteristics specific to mixers. 
Regular transactions ("other") have a more natural and spread-out distribution, while mixers like Wasabi and JoinMarket form tighter clusters in a lower-entropy region.

<!-- Original notebook cell 33 -->

### 2.4 Correlation Matrix

![Feature correlations](../assets/original/cell-34-figure-1.png)

*Feature correlations. Original saved output, cell 34.*

<!-- Original notebook cell 35 -->

The correlation matrix highlights several expected redundancies and validates the independence of our behavioral features. 
We can notice that features like n_inputs and vsize are linked, as more inputs naturally increase the transaction size.
However, our main key features are independant (not very linked) that ensure unique information from each feature and better analysis for next steps.

<!-- Original notebook cell 36 -->

## Model Training and Evaluation

<!-- Original notebook cell 37 -->

In this part, I decided to focus exclusively on XGBoost; it is one of the best algorithms for classifying tabular data into different classes and is, above all, very easy to interpret, notably thanks to SHAP. 

In the first part, I verify that the feature extraction went well by performing cross-validation; the goal of this preliminary cross-validation is to have a relatively constant F1-score across the folds to ensure that the extraction had no issues. 

I also take this opportunity to remove features that were too highly correlated or those that only represented overhead and skewed my results later on. 
For example, I had tried to create a mixture of features, but it only created noise. I therefore removed them, but I am leaving them in the code for informational purposes.

For the remainder of the problem, I used data stratification to ensure a good distribution given the low number of mixers. 

Finally, I applied sample weights so that the algorithm penalizes errors on minority classes more heavily, guaranteeing that the F1-score reflects a real detection capability for minority classes and not just the majority class ('other').

<!-- Original notebook cell 38 -->

### 3.1 Cross-Validation Test

<!-- Original notebook cell 40 -->

### 3.2 Implementation and training of the model

<!-- Original notebook cell 42 -->

To try to converge toward the best solution, I implemented a hyperparameter search that I conducted in several stages, and finally, I focused the search among the best candidates I had previously obtained.

Furthermore, I use the early_stopping_rounds parameter to stop the model as soon as it stops learning and begins to overfit.

<!-- Original notebook cell 45 -->

### 3.3 Analysis of the model : Confusion Matrix & ROC Curve

![Normalised confusion matrix](../assets/original/cell-47-figure-1.png)

*Normalised confusion matrix. Original saved output, cell 47.*

![One-vs-rest ROC curves](../assets/original/cell-48-figure-1.png)

*One-vs-rest ROC curves. Original saved output, cell 48.*

<!-- Original notebook cell 49 -->

The model seems overall good with an average AUC of 0.9981 and an F1-score of 0.95. The model successfully identifies almost all labels, however, the one it struggles with the most is the Wasabi mixer, which it misclassifies as 'other' about once every ten times. 

I believe I have already tested enough parameters, etc., to reach a fairly optimized model, and I think that the limit of my model here lies in my features and my knowledge of blockchain and CoinJoin. I tried to research and find features that could specifically detect Wasabi transactions, but it only deteriorated my model, so I kept this one.

*Note: I had sent you an email regarding the paths I had followed, but I decided to submit everything in a single notebook because the results were far too inferior to this one. But overall, I had tried using the Random Forest model and the VotingClassifier library so that my model would vote between RF and XGB each time, but this deteriorated my model. 
I also tested the StackingClassifier library to once again compare RF and XGB, but I had difficulty tuning the model that was supposed to choose between the two, and the results were very poor as well.
The last path I explored was to find the backbone of my model with the best parameters and then significantly increase the number of trees while gradually reducing the learning rate, but I encountered a fairly counter-intuitive result: it was not necessarily the lowest learning rate that yielded the best results.*

<!-- Original notebook cell 50 -->

## 4. Model explainability

<!-- Original notebook cell 51 -->

### 4.1 SHAP

![Global SHAP importance](../assets/original/cell-52-figure-1.png)

*Global SHAP importance. Original saved output, cell 52.*

<!-- Original notebook cell 53 -->

### 4.2 SHAP per Class

![SHAP: Bitcoin Fog](../assets/original/cell-54-figure-1.png)

*SHAP: Bitcoin Fog. Original saved output, cell 54.*

![SHAP: JoinMarket](../assets/original/cell-54-figure-2.png)

*SHAP: JoinMarket. Original saved output, cell 54.*

![SHAP: SamouraiWallet.com](../assets/original/cell-54-figure-3.png)

*SHAP: SamouraiWallet.com. Original saved output, cell 54.*

![SHAP: WasabiWallet.io](../assets/original/cell-54-figure-4.png)

*SHAP: WasabiWallet.io. Original saved output, cell 54.*

![SHAP: other](../assets/original/cell-54-figure-5.png)

*SHAP: other. Original saved output, cell 54.*

<!-- Original notebook cell 55 -->

The SHAP analysis really surprised me, I did not expect to see certain values so high or low. 

Primarily the entropy, before this, I thought it was one of my best features because I believed it would perform well, especially in distinguishing between a mixer and a normal transaction, but it is apparently not as significant. 

On the other hand, the n_locktime feature is the most discriminating, especially for Bitcoin Fog and JoinMarket, which proves that these two mixers may still rely on legacy code and do not update their nLockTime. 

I also separately represented the feature importance for each label, here, we can clearly see the features that manage to categorize to the right or left (mixer/other) depending on their value

<!-- Original notebook cell 56 -->

## 5. Validation set predictions

