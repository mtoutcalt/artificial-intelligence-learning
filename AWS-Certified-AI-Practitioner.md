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

---
 
### Q12 — Types of ML
 
A robotics company trains an AI to navigate a warehouse by rewarding it every time it successfully delivers a package and penalizing it when it collides with shelves. Which type of ML is this?
 
- A) Supervised learning
- B) Unsupervised learning
- C) Reinforcement learning
- D) Self-supervised learning
**✅ Answer: C — Reinforcement learning**
 
The key signals are *rewards, penalties, and an agent making decisions*. The model isn't learning from labeled examples — it's learning through trial and error in an environment. Watch for trigger phrases:
- "rewards" / "penalties" / "agent" → Reinforcement learning
---
 
### Q13 — Bias in Training Data
 
A company trains a loan approval model on 10 years of historical decisions. It later discovers the model approves loans at a significantly lower rate for certain demographic groups. What is the most likely root cause?
 
- A) Overfitting to recent data
- B) Model drift
- C) Bias in the training data
- D) Underfitting due to too few features
**✅ Answer: C — Bias in the training data**
 
Historical data often reflects past human biases. When a model trains on biased data, it learns and perpetuates those biases. This is one of the core responsible AI concerns on the exam. Always ask: *"Where did this data come from, and who made the original decisions?"*
 
---
 
### Q14 — Training vs Inference
 
After a company finishes training a model that predicts equipment failures, they deploy it to a factory floor where it analyzes live sensor data every 30 seconds. What is this deployed phase called?
 
- A) Training
- B) Fine-tuning
- C) Inference
- D) Evaluation
**✅ Answer: C — Inference**
 
*Training* is when the model learns from data. *Inference* is when the trained model is used to make predictions on new, real-world data. The exam often tests whether you can distinguish these two phases.
 
---
 
### Q15 — Feature Engineering
 
A data scientist is building a model to predict flight delays. The raw dataset includes a "departure timestamp" column. She creates three new columns from it: hour of day, day of week, and month. What technique is she applying?
 
- A) Data augmentation
- B) Regularization
- C) Feature engineering
- D) Hyperparameter tuning
**✅ Answer: C — Feature engineering**
 
Feature engineering is the process of transforming raw data into more useful inputs for a model. Extracting hour, day, and month from a timestamp is a classic example — the model can now learn patterns like "Friday evening flights delay more."
 
**Quick eliminations:**
- **Augmentation** — creating variations of existing data (mainly used for images)
- **Regularization** — a technique to reduce overfitting by penalizing model complexity
- **Hyperparameter tuning** — adjusting settings before training begins
---
 
### Q16 — Hyperparameters vs Parameters
 
A machine learning engineer is tuning her neural network before training begins. She experiments with different learning rates, batch sizes, and numbers of hidden layers to improve performance. What is she adjusting?
 
- A) Model parameters
- B) Training labels
- C) Hyperparameters
- D) Feature weights
**✅ Answer: C — Hyperparameters**
 
*Parameters* (like weights and biases) are learned automatically by the model during training. *Hyperparameters* are set by the engineer *before* training and control how the training process itself works.
 
| | Parameters | Hyperparameters |
|---|---|---|
| Set by | Model (learned during training) | Engineer (set before training) |
| Examples | Weights, biases | Learning rate, batch size, epochs |
| Adjusted via | Backpropagation | Manual tuning / AutoML |
 
---
 
### Q17 — Overfitting
 
A model's performance on training data is great, but it fails on new data. You add more training examples and the gap closes. What was the original problem?
 
- A) Underfitting
- B) Model drift
- C) Overfitting
- D) Data leakage
**✅ Answer: C — Overfitting**
 
A large gap between training and test performance is the classic overfitting signal. Adding more training data is one of the standard fixes — it forces the model to learn generalizable patterns rather than memorizing examples.
 
---
 
### Q18 — Unsupervised Learning
 
A recommendation system learns that users who buy hiking boots often buy wool socks, without being told to look for that relationship. What type of ML is this?
 
- A) Supervised learning
- B) Reinforcement learning
- C) Deep learning
- D) Unsupervised learning
**✅ Answer: D — Unsupervised learning**
 
No labels, no predefined answers — the model is discovering hidden structure and relationships in the data on its own. Association rule learning (finding purchase patterns) is a classic unsupervised technique.
 
---
 
### Q19 — Reinforcement Learning
 
A self-driving car AI receives a +10 score for reaching its destination safely and a -5 score for every traffic violation. What type of ML is being used?
 
- A) Supervised learning
- B) Unsupervised learning
- C) Reinforcement learning
- D) Semi-supervised learning
**✅ Answer: C — Reinforcement learning**
 
Scores, rewards, and penalties given to an agent navigating an environment are the defining characteristics of reinforcement learning.
 
---
 
### Q20 — Evaluation Metrics: Recall
 
A cancer screening model catches 95% of all real cancer cases but also flags many healthy patients as potentially cancerous. Which metric is high?
 
- A) Precision
- B) Accuracy
- C) F1 Score
- D) Recall
**✅ Answer: D — Recall**
 
The model is catching 95% of all actual cancer cases — that is the definition of high recall (TP / (TP + FN)). The fact that it also flags many healthy patients means precision is low, but the question asks only what is *high*.
 
---
 
### Q21 — Model Drift
 
A model was highly accurate last year but has degraded steadily over 6 months as user behavior has shifted. What is the most likely cause?
 
- A) Overfitting
- B) Data leakage
- C) Underfitting
- D) Model drift
**✅ Answer: D — Model drift**
 
When the real world changes but the model doesn't, performance degrades over time. Gradual decline in accuracy following a behavioral or environmental shift is the key signal for model drift.
 
