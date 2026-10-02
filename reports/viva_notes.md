# Viva Notes — Explainable Student Performance Prediction

> Prepared for college mini-project presentation and examiner Q&A.

---

## 1. Simple Explanations of the Core Concepts

### Machine Learning Models Used

**Ridge Regression (Linear Model)**
- Fits a straight line (hyperplane in multiple dimensions) through the data.
- Adds an L2 penalty (`alpha × sum of squared coefficients`) to prevent overfitting.
- Pros: Interpretable coefficients, fast, no overfitting on small datasets.
- Cons: Cannot capture non-linear patterns or interaction effects (e.g., "attendance helps more when combined with high study hours").
- *Analogy*: Like drawing the best single straight line through a scatter plot.

**Random Forest (Ensemble — Bagging)**
- Trains hundreds of independent decision trees, each on a random bootstrap sample of data and a random subset of features.
- Final prediction = average of all trees' predictions.
- The randomness makes trees diverse; averaging cancels out their individual errors.
- Pros: Handles non-linear interactions, robust to outliers, produces reliable SHAP values.
- Cons: Slower to train, less interpretable than a single tree.
- *Analogy*: Asking 100 teachers independently to grade a student and averaging their scores.

**XGBoost (Ensemble — Gradient Boosting)**
- Trains trees *sequentially*, where each tree corrects the residual errors of the previous tree.
- Uses second-order gradient information (Newton steps) to converge faster.
- Pros: State-of-the-art on tabular data, regularisation built in.
- Cons: More hyperparameters to tune, can overfit if not constrained.
- *Analogy*: A student who reviews their wrong answers each attempt and specifically studies only the topics they got wrong.

---

### SHAP (SHapley Additive exPlanations)

**What it is:**
SHAP is a principled, game-theoretic method for attributing each feature's contribution to a single prediction. It is based on *Shapley values* from cooperative game theory.

**Core Idea:**
For a given student, SHAP answers: *"How much did Attendance_Rate specifically contribute to this student's predicted score, compared to what the average student gets?"*

**Formula (simplified):**
```
Prediction = Base Value (average prediction) + sum of all SHAP values
50.18 = 58.85 + (−4.05) + (−3.02) + (−1.70) + 0.08 + 0.06 + 0.01
```

**TreeExplainer:**
- Specialized for tree models (Random Forest, XGBoost).
- Computes exact Shapley values in polynomial time by exploiting tree structure.
- Much faster than the general KernelExplainer.

**Properties guaranteed by Shapley values:**
1. *Efficiency* — SHAP values sum exactly to the prediction gap from the baseline.
2. *Symmetry* — Features with identical contributions receive identical values.
3. *Dummy* — Features that don't change the prediction receive SHAP = 0.
4. *Additivity* — The contributions of two independent models sum correctly.

**Why 1:1 column mapping matters:**
Each original feature must map to exactly one processed column. If `Gender` were one-hot encoded into `Gender_Male` / `Gender_Female`, the SHAP value would split across two columns — making it impossible to say "Gender contributed X to this prediction" as a single number.

---

## 2. Ten Likely Examiner Questions with Short Answers

**Q1: Why did you choose Random Forest over XGBoost when XGBoost usually wins Kaggle competitions?**

*A:* On this specific dataset (708 students, 7 features), Random Forest produced lower test RMSE (2.97 vs 3.32). XGBoost's sequential boosting approach can overfit when the dataset is small and features are few. Random Forest's bagging with 100 trees was sufficient to capture the non-linear interactions present. We validated this on a held-out test set, not just cross-validation.

---

**Q2: What exactly is data leakage and how did you prevent it?**

*A:* Data leakage occurs when information from the test set "leaks" into training — for example, fitting a StandardScaler on the entire dataset before splitting would encode test-set statistics into the scaler. We prevented this strictly: the train/test split is performed *first* (80/20, `random_state=42`), and all imputers, encoders, and scalers are fit *exclusively* on the training split and then applied to the test split. This is enforced via scikit-learn `Pipeline` and `ColumnTransformer`.

---

**Q3: Why use ordinal encoding instead of one-hot encoding for categorical variables?**

*A:* Two reasons. First, for *ordinal* features like `Parental_Education_Level` (High School < Bachelors < Masters < PhD), ordinal encoding preserves meaningful rank information that one-hot encoding discards. Second, the project rule requires 1:1 feature mapping for SHAP interpretability — one-hot would split each category into separate dummy columns, making it impossible to show a single SHAP value for "Parental Education Level" in the explanation.

---

**Q4: What does a SHAP value of −4.05 for Attendance_Rate mean?**

*A:* It means that this student's attendance rate (54.3%) pulled their predicted score *4.05 points below* the average prediction of 58.85. The baseline of 58.85 is the model's prediction if it knew nothing about this specific student. The SHAP value decomposes the gap: attendance alone explains −4.05 of the total deviation.

---

**Q5: Why is the LLM forbidden from deciding reasons itself?**

*A:* Because LLMs can hallucinate confident-sounding but factually incorrect explanations. If the model was allowed to reason freely, it might say "the student likely fails because they come from a low-income family" — which is (a) not in the data payload, (b) causal and potentially stigmatizing, and (c) unverifiable. The SHAP payload acts as a strict factual grounding: the LLM only rephrases pre-computed, model-derived attributions.

---

**Q6: Why is the Fail class empty under the default threshold (< 50)?**

*A:* The dataset's exam scores range from 50 to 77; no student scores below 50. This is a strong signal that the data is *synthetic* — real student score distributions typically have a long left tail with genuine failures. Under the strict rule `Fail < 50`, the Fail bucket is always empty. We identified this in the threshold sensitivity analysis and recommended `Fail < 55` as a calibrated threshold that captures 47 at-risk students (33% of the test set) with 78.7% recall.

---

**Q7: What is the difference between global and local SHAP explanations?**

*A:* **Global SHAP** (beeswarm/bar chart over the entire test set) shows which features most influence predictions *on average* across all students — useful for model understanding. **Local SHAP** (waterfall chart for one student) shows the exact contribution of each feature for *that specific student's* prediction. The Streamlit app shows local SHAP — because it's answering "why did *this* student get *this* score?", not "what matters in general?"

---

**Q8: Your model achieves R² = 0.793. What does that mean?**

*A:* R² measures the fraction of variance in `Final_Exam_Score` that the model explains. 0.793 means our Random Forest explains ~79.3% of the variation in test-set exam scores using the 7 available features. The remaining ~20.7% is variance not captured by the features — it could be due to unmeasured factors like student mood on exam day, exam difficulty variation, or measurement noise.

---

**Q9: How do you know the LLM isn't hallucinating?**

*A:* We can't be 100% certain, but we built an automated evaluation (`src/evaluate.py`) that checked 50 explanations against four rules:
1. Factor Recall — does the text mention the features listed in the SHAP payload?
2. Hallucination check — does the text mention any feature NOT in the payload?
3. Forbidden content — does the advice section mention gender/income/school type?
4. Causal language — do phrases like "because of" or "due to" appear?
Across 50 students, the LLM achieved 0% hallucinations, 0% forbidden advice, and 0% causal language violations. Factor recall was 78.4%.

---

**Q10: What would you improve with more time?**

*A:* See the Future Work section below, but the top three improvements are: (1) Train on a verified real-world student dataset to remove the synthetic-data limitation. (2) Implement counterfactual explanations (DiCE) to show the *minimum change* needed for a student to move from Fail to Pass. (3) Conduct a user study with actual students and teachers to measure whether the plain-English explanations are actionable and understood.

---

## 3. Future Work

| # | Area | Description |
|---|---|---|
| 1 | **Counterfactual Explanations (DiCE)** | Use the DiCE library to generate the minimum-cost feature changes needed to cross the Pass boundary. Unlike SHAP (which explains the current state), DiCE answers "what would need to change to get a Pass?" — more directly actionable for students. |
| 2 | **RAG for Personalized Study Tips** | Build a Retrieval-Augmented Generation (RAG) pipeline over a knowledge base of pedagogical research and study-skills literature. When the LLM suggests "increase study hours", it could cite specific evidence-based strategies retrieved from the knowledge base. |
| 3 | **Local LLM Comparison (Ollama / Mistral)** | Compare the proprietary Gemini layer against a locally-hosted open-source model (e.g., Mistral-7B via Ollama) on the same 50-student faithfulness benchmark. This would quantify the trade-off between API dependency, cost, privacy, and explanation quality. |
| 4 | **User Study** | Conduct a structured user study with actual students, teachers, and parents. Collect Likert-scale ratings on clarity, trust, and usefulness of explanations. Measure whether the interventions suggested by the what-if simulator translate into behavioral changes. |
| 5 | **Real Dataset Validation** | Replicate the entire pipeline on a verified, non-synthetic student dataset (e.g., UCI Student Performance Dataset by Paulo Cortez, or a partner institution's anonymized records) to test generalizability. |
| 6 | **Conformal Prediction Intervals** | Replace point predictions with calibrated confidence intervals (e.g., using MAPIE) so that the app communicates "score will likely be between 56 and 63 with 90% confidence" — more honest than a single number. |
| 7 | **Multi-Task Output** | Jointly predict both `Final_Exam_Score` (regression) and `Pass_Fail` (classification) with a shared representation, potentially improving both tasks through shared feature learning. |
| 8 | **Longitudinal Tracking** | Extend to a semester-level dashboard where each student's historical predictions, attended interventions, and actual scores are tracked over time — transforming the tool from a one-shot predictor into a continuous learning companion. |

---

*Generated as part of: Explainable Student Performance Prediction (College Mini-Project, 2026)*
