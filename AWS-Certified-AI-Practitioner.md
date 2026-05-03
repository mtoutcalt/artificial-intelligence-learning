# AWS Certified AI Practitioner — Domain 1 Practice Questions
## Fundamentals of AI and ML

---

### Q1 — AI vs ML vs Deep Learning

A company wants to build a system that automatically detects fraudulent transactions by learning patterns from millions of historical transactions, without engineers manually writing the fraud detection rules. Which type of technology is this an example of?

- A) Traditional rule-based programming
- B) Machine Learning
- C) Deep Learning
- D) Expert Systems

**✅ Answer: B — Machine Learning**

The key signal is *"learning patterns from data"* without manually written rules. Deep learning could work here, but ML is the better general answer since the question gives no signal that deep neural networks are needed.

---

### Q2 — Types of ML

A retail company groups its customers into segments based on purchasing behavior, without telling the model in advance what the groups should be. Which type of ML is this?

- A) Supervised learning
- B) Unsupervised learning
- C) Reinforcement learning
- D) Deep learning

**✅ Answer: B — Unsupervised learning**

No labels, no predefined answers — the model is discovering structure on its own. Watch for trigger phrases:
- "labeled data" / "correct answers provided" → Supervised
- "find hidden patterns" / "no predefined categories" → Unsupervised
- "trial and error" / "rewards and penalties" / "agent" → Reinforcement

---

### Q3 — ML Lifecycle / Model Drift

After deploying an ML model to production, a data scientist notices its accuracy has steadily dropped over the past 3 months because customer behavior has changed significantly. What is this phenomenon called?

- A) Overfitting
- B) Underfitting
- C) Model drift
- D) Data leakage

**✅ Answer: C — Model drift**

When the real world changes but your model doesn't, performance degrades over time. This is why the Monitor step of the ML lifecycle exists (e.g. SageMaker Model Monitor).

- **Overfitting** — model memorizes training data, performs poorly on new data
- **Underfitting** — model too simple, performs poorly even on training data
- **Data leakage** — test/future data accidentally leaks into training, giving falsely inflated results

---

### Q4 — Overfitting vs Underfitting

A model performs extremely well on training data with 99% accuracy, but only achieves 62% accuracy on new unseen data. What is the most likely cause, and what is the recommended fix?

- A) Underfitting — add more features
- B) Overfitting — use more training data or apply regularization
- C) Model drift — retrain the model on recent data
- D) Data leakage — remove correlated features

**✅ Answer: B — Overfitting**

A huge gap between training accuracy (99%) and test accuracy (62%) is the classic overfitting signal. The model memorized training data instead of learning generalizable patterns.

**Fixes for overfitting:**
- More training data
- Regularization (L1/L2)
- Dropout (neural networks)
- Simplify the model

---

### Q5 — Data Augmentation

A data scientist is training an image classification model but only has 2,000 labeled photos. The model is overfitting badly. She doesn't have budget to collect more real photos. Which technique would most directly address this without collecting new data?

- A) L2 Regularization
- B) Data augmentation
- C) Dropout
- D) Feature selection

**✅ Answer: B — Data augmentation**

Augmentation artificially expands your dataset by creating variations of existing data (flipping, rotating, cropping images). The key constraint was "without collecting new data."

---

### Q6 — Overfitting vs Underfitting (Subtle)

A team trains a logistic regression model to predict customer churn. It performs similarly on both training data (78% accuracy) and test data (76% accuracy), but the business considers this accuracy too low to be useful. What is the most likely problem?

- A) Overfitting — the model is too complex
- B) Model drift — the data has changed over time
- C) Underfitting — the model is too simple to capture the patterns
- D) Data leakage — the test data contaminated the training process

**✅ Answer: C — Underfitting**

Similar performance on training and test data (small gap) but both are low = underfitting. The model is consistently mediocre.

| Situation | Training Accuracy | Test Accuracy | Problem |
|---|---|---|---|
| High gap | High (e.g. 99%) | Low (e.g. 62%) | Overfitting |
| No gap, both low | Low (e.g. 78%) | Low (e.g. 76%) | Underfitting |
| Gradually declining | Was good | Dropping over time | Model drift |
| Suspiciously perfect | Unrealistically high | Unrealistically high | Data leakage |

---

### Q7 — Well-Fitted Model

An ML model predicts house prices. On training data it achieves a mean error of $8,000. On brand new listings it achieves a mean error of $9,200. The business is happy with this performance. What best describes this model?

- A) Overfitting
- B) Underfitting
- C) Well-fitted
- D) Data leakage

**✅ Answer: C — Well-fitted**

Small gap between training and test performance, business is satisfied. The model generalizes well without memorizing training data.

---

### Q8 — Evaluation Metrics: Recall

A hospital builds a model to detect cancer from scans. A false negative means the model misses a real cancer case. The team wants to minimize missed diagnoses above all else. Which metric should they prioritize?

- A) Precision
- B) Accuracy
- C) Recall
- D) F1 Score

**✅ Answer: C — Recall**

Missing a real cancer case (false negative) is catastrophic. Recall measures how many actual positives the model caught.

| Metric | Question it answers | Use when... |
|---|---|---|
| Accuracy | How often am I right overall? | Data is balanced |
| Precision | When I predict positive, am I right? | False alarms are costly |
| Recall | Did I catch all the real positives? | Missing cases is costly |
| F1 Score | Am I good at both? | Imbalanced data, need balance |

---

### Q9 — Evaluation Metrics: Precision

A bank's fraud detection model flags legitimate transactions as fraudulent, causing customers to have their cards blocked unnecessarily. Which metric should the team focus on improving?

- A) Recall
- B) Precision
- C) F1 Score
- D) Accuracy

**✅ Answer: B — Precision**

The problem is too many false positives (flagging legitimate transactions). Precision measures the quality of positive predictions — improving it reduces false alarms.

---

### Q10 — Evaluation Metrics: Recall

A hiring tool screens resumes and is being evaluated. The company discovers it's rejecting many qualified candidates. Which metric is most likely low?

- A) Precision
- B) Accuracy
- C) F1 Score
- D) Recall

**✅ Answer: D — Recall**

Qualified candidates are being missed = false negatives. Low recall means the model isn't catching all the real positives (qualified candidates).

---

### Q11 — Checkpoint (Multi-Answer)

A company trains a model to detect defective products on a factory assembly line. Only 1% of products are actually defective. The model achieves 99% accuracy by predicting "not defective" for every single product. Which TWO statements are true?

- A) The model is overfitting
- B) The model has a recall of 0%
- C) Accuracy is a misleading metric here due to class imbalance
- D) The model is underfitting
- E) Precision and recall should both be prioritized equally

**✅ Answer: B and C**


- **C** — Only 1% of products are defective, so a lazy model that always says "not defective" scores 99% accuracy without learning anything. Classic imbalanced data trap.
- **B** — The model never predicts defective, so it catches zero actual defects. Recall = TP / (TP + FN) = 0 / (0 + 100) = 0%.

