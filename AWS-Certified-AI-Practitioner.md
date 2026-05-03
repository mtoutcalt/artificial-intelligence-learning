# AWS Certified AI Practitioner — Practice Questions
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

---

### Q22 — Data Splitting

A data scientist splits her dataset into two parts: 80% for training and 20% for testing. She trains several models, compares them on the test set, and picks the best one. A senior engineer warns her this approach is flawed. What is the correct fix?

- A) Use all the data for training to maximize model performance
- B) Add a separate validation set so the test set is never used during model selection
- C) Increase the training split to 95% to reduce overfitting
- D) Apply data augmentation before splitting

**✅ Answer: B — Add a separate validation set**

Every time she compares models on the test set, the test set is influencing her decisions — it's no longer a true blind evaluation.

**The correct 3-way split:**

| Split | Purpose |
|---|---|
| **Training set** | Model learns from this |
| **Validation set** | You compare & tune models on this |
| **Test set** | Final, one-time blind evaluation only |

Think of the test set like a final exam — if you keep peeking at it while studying, your score doesn't mean anything anymore.

---

### Q23 — Transfer Learning

A startup wants to build an image recognition model to identify rare bird species. They only have 500 labeled photos. A researcher suggests using a model already trained on millions of general images and adapting it for their task. What technique is this?

- A) Reinforcement learning
- B) Data augmentation
- C) Transfer learning
- D) Semi-supervised learning

**✅ Answer: C — Transfer learning**

The key phrase is *"already trained on millions of general images and adapting it for their task."* Transfer learning takes knowledge a model built up on one task and repurposes it for another.

It solves two problems at once:
- **Small dataset problem** — 500 photos isn't enough to train from scratch, but the model already "knows" what edges, shapes, and textures look like
- **Cost/time problem** — training from scratch on millions of images is expensive; you're skipping that

---

### Q24 — Regression vs Classification

A model is trained to predict the exact sale price of a house given its features (size, location, age). Which type of ML task is this?

- A) Binary classification
- B) Multi-class classification
- C) Clustering
- D) Regression

**✅ Answer: D — Regression**

The giveaway is *"predict the exact sale price"* — any time a model outputs a continuous number, that's regression.

| Task | Output | Example |
|---|---|---|
| **Regression** | A number on a scale | House price: $347,000 |
| **Binary classification** | One of two categories | Spam or not spam |
| **Multi-class classification** | One of many categories | Cat, dog, or bird |
| **Clustering** | Groups discovered by the model | Customer segments |

**Simple rule:** Ask yourself *"is the answer a number or a category?"* Number → regression. Category → classification.

---

### Q25 — Responsible AI

A company deploys a hiring algorithm and later learns it performs significantly worse for candidates from certain universities because those schools were underrepresented in the training data. Which responsible AI principle is most directly violated?

- A) Explainability
- B) Fairness
- C) Robustness
- D) Privacy

**✅ Answer: B — Fairness**

The model performing *worse for a specific group* due to underrepresentation in training data is a textbook fairness violation.

| Principle | What it means | Example violation |
|---|---|---|
| **Fairness** | Model performs equitably across groups | Hiring model disadvantages certain candidates |
| **Explainability** | You can understand *why* the model made a decision | A loan denial with no reason given |
| **Robustness** | Model performs reliably under unexpected conditions | Model breaks when given slightly noisy data |
| **Privacy** | Personal data is protected | Training data exposes sensitive user info |

Fairness issues almost always trace back to the training data — either a group is underrepresented, or the historical data itself reflected human bias.

---

### Q26 — Batch vs Real-Time Inference

A bank wants to flag fraudulent transactions *as they happen* so cards can be blocked immediately. Another bank runs fraud checks nightly on the previous day's transactions. Which inference type does each bank use, respectively?

- A) Batch inference / Real-time inference
- B) Real-time inference / Batch inference
- C) Both use real-time inference
- D) Both use batch inference

**✅ Answer: B — Real-time inference / Batch inference**

| Type | When it runs | Latency | Example |
|---|---|---|---|
| **Real-time inference** | Instantly, as data arrives | Milliseconds | Fraud detection, autocomplete |
| **Batch inference** | On a schedule, on accumulated data | Minutes/hours | Nightly reports, monthly credit scoring |

**Giveaway phrases:**
- *"as it happens," "immediately," "live"* → Real-time
- *"nightly," "weekly," "end of day," "accumulated"* → Batch

---

### Q27 — Generative AI

A developer uses an AI system to generate entirely new product descriptions by typing a plain English instruction like *"write a playful description for a kids' backpack."* The system produces original text it was never explicitly shown before. What best describes this type of AI?

- A) Discriminative AI
- B) Supervised learning
- C) Generative AI
- D) Reinforcement learning

**✅ Answer: C — Generative AI**

The giveaway is *"generates entirely new content"* from a plain English instruction.

| Type | What it does | Example |
|---|---|---|
| **Generative AI** | Creates new content (text, images, audio) | Writing product descriptions, generating images |
| **Discriminative AI** | Classifies or labels existing data | Spam filter, fraud detection, image classification |

- Discriminative = *draws a boundary* between things
- Generative = *creates something new*

---

### Q28 — Prompt Engineering

A developer is working with a large language model and wants it to answer customer service questions in a formal, professional tone. Instead of retraining the model, she experiments with different ways of wording her instructions to get better outputs. What technique is she using?

- A) Fine-tuning
- B) Transfer learning
- C) Prompt engineering
- D) Hyperparameter tuning

**✅ Answer: C — Prompt engineering**

The key phrase is *"instead of retraining the model"* — she's getting better results purely by changing how she words her instructions.

| Technique | Changes the model? | How it works | Cost |
|---|---|---|---|
| **Prompt engineering** | No | Craft better inputs | Very cheap |
| **Fine-tuning** | Yes | Retrain on new domain-specific data | Moderate |
| **Transfer learning** | Yes | Adapt a pre-trained model to a new task | High |

**Common prompt engineering techniques:**
- **Zero-shot** — just give the instruction, no examples
- **Few-shot** — include a few examples in the prompt to guide the model
- **Chain of thought** — ask the model to reason step by step

---

### Q29 — Foundation Models

A company wants to build a customer support chatbot. Instead of training a model from scratch, they access a large pre-trained model via API and adapt it to their use case. What term best describes the large pre-trained model they are using?

- A) A supervised learning model
- B) A foundation model
- C) A clustering model
- D) A reinforcement learning agent

**✅ Answer: B — A foundation model**

| Characteristic | Detail |
|---|---|
| **Size** | Trained on massive amounts of data |
| **General purpose** | Can handle text, images, code and more |
| **Adaptable** | Can be fine-tuned or prompted for specific tasks |
| **Examples** | Claude, GPT-4, Llama, Amazon Titan |

Instead of every company training their own model from scratch (extremely expensive), foundation models are built once and reused by many — you just adapt them via prompting or fine-tuning.

---

### Q30 — Retrieval Augmented Generation (RAG)

A company's chatbot keeps giving outdated answers about their products because the foundation model it uses was trained on data from two years ago. A developer suggests connecting the model to the company's internal knowledge base so it can look up current information before responding. What technique is this?

- A) Fine-tuning
- B) Prompt engineering
- C) Retrieval Augmented Generation (RAG)
- D) Transfer learning

**✅ Answer: C — Retrieval Augmented Generation (RAG)**

The giveaway is *"connecting the model to an external knowledge base to look up current information."*

**How RAG works:**
1. User asks a question
2. System searches the knowledge base for relevant information
3. That information is injected into the prompt
4. The model answers using both its training AND the retrieved info

| | RAG | Fine-tuning |
|---|---|---|
| **How it works** | Connects model to external data at query time | Retrains model on new data |
| **Best for** | Current, frequently changing information | Teaching the model a new style or domain |
| **Cost** | Cheaper | More expensive |
| **Data stays fresh?** | Yes, update the knowledge base anytime | No, model knowledge is frozen after training |

---

### Q31 — Hallucination

A customer service chatbot built on a foundation model confidently tells a user that a product comes with a 5 year warranty. The company never offered this warranty — the model simply made it up and presented it as fact. What is this phenomenon called?

- A) Model drift
- B) Overfitting
- C) Hallucination
- D) Data leakage

**✅ Answer: C — Hallucination**

Foundation models are trained to produce fluent, confident-sounding responses. They don't have a built-in "I don't know" mechanism — they'll fill gaps in their knowledge with plausible-sounding but completely fabricated information.

**How to reduce hallucinations:**

| Technique | How it helps |
|---|---|
| **RAG** | Grounds responses in real, retrieved facts |
| **Prompt engineering** | Instruct the model to say "I don't know" when uncertain |
| **Fine-tuning** | Train on domain-specific accurate data |
| **Human review** | Catch errors before they reach users |

---

### Q32 — Amazon Bedrock: Knowledge Bases

A company wants to use Amazon Bedrock to build a chatbot that can answer questions about their internal HR policies. Their documents are stored in Amazon S3 and are updated monthly. They want the chatbot to always have access to the latest versions without retraining. Which Bedrock feature should they use?

- A) Bedrock Fine-tuning
- B) Bedrock Knowledge Bases
- C) Bedrock Agents
- D) Prompt engineering

**✅ Answer: B — Bedrock Knowledge Bases**

The giveaways are *"latest versions without retraining"* and *"documents stored in S3"* — that's RAG in action, and Knowledge Bases is Bedrock's built-in RAG feature.

| | Knowledge Bases | Fine-tuning |
|---|---|---|
| **Monthly doc updates** | Just update S3, done | Retrain every month, expensive |
| **Always current** | Yes | No |
| **Cost** | Low | High |

---

### Q33 — Amazon Bedrock: Agents

A travel company builds an AI assistant on Amazon Bedrock that can check flight availability, book tickets, and send confirmation emails — all in response to a single user request. Which Bedrock feature makes this possible?

- A) Bedrock Knowledge Bases
- B) Bedrock Fine-tuning
- C) Bedrock Agents
- D) Prompt engineering

**✅ Answer: C — Bedrock Agents**

The giveaway is the model *taking multiple actions* in sequence — checking availability, booking, then emailing.

| | Regular model | Bedrock Agent |
|---|---|---|
| **What it does** | Answers questions | Takes actions in the world |
| **Multi-step?** | No | Yes |
| **Can call APIs?** | No | Yes |
| **Example** | "What flights exist?" | "Find, book, and confirm a flight" |

Think of a regular model as a very smart person who can only talk. An agent is that same person but they can also pick up the phone, open a browser, and send emails.

---

### Q34 — Amazon Bedrock: Prompt Engineering vs Fine-tuning

A company is using Amazon Bedrock to build a legal document summarizer. The foundation model keeps responding in a casual, conversational tone which isn't appropriate for their lawyers. They want the model to always respond formally without changing the underlying model. What is the cheapest and fastest solution?

- A) Fine-tune the model on formal legal documents
- B) Use a different foundation model
- C) Use prompt engineering to instruct the model to respond formally
- D) Build a Bedrock Agent to reformat responses

**✅ Answer: C — Prompt engineering**

The giveaways are *"without changing the underlying model"* and *"cheapest and fastest."*

**The cost/speed hierarchy:**

| Approach | Cost | Speed | Changes model? |
|---|---|---|---|
| **Prompt engineering** | $ | Minutes | No |
| **RAG / Knowledge Bases** | $$ | Hours | No |
| **Fine-tuning** | $$$ | Days | Yes |

When a question says "without retraining" or "cheapest solution," work left to right on that table.

---

### Q35 — Amazon Bedrock: Model Selection

A startup is evaluating which foundation model to use on Amazon Bedrock. They need to generate photorealistic images from text descriptions for their interior design app. Which model family on Bedrock is most appropriate?

- A) Amazon Titan
- B) Anthropic Claude
- C) Stability AI Stable Diffusion
- D) Meta Llama

**✅ Answer: C — Stability AI Stable Diffusion**

The giveaway is *"photorealistic images from text descriptions"* — Stable Diffusion is specifically an image generation model.

| Model | Provider | Best for |
|---|---|---|
| **Claude** | Anthropic | Text generation, reasoning, summarization |
| **Titan** | Amazon | Text generation, embeddings, image generation |
| **Llama** | Meta | Text generation, open source flexibility |
| **Mistral** | Mistral | Text generation, lightweight and fast |
| **Stable Diffusion** | Stability AI | Image generation from text |

**Rule of thumb:**
- Need to generate or understand **text**? → Claude, Titan, Llama, or Mistral
- Need to generate **images**? → Stable Diffusion or Titan Image Generator

---

### Q36 — Amazon Bedrock: Guardrails

A financial services company is deploying a customer facing chatbot on Amazon Bedrock. They are worried users might try to trick the model into giving investment advice, making politically controversial statements, or revealing sensitive competitor information. Which Bedrock feature should they use to prevent this?

- A) Bedrock Agents
- B) Bedrock Guardrails
- C) Bedrock Knowledge Bases
- D) Fine-tuning

**✅ Answer: B — Bedrock Guardrails**

The giveaway is the need to *prevent* specific types of harmful or off-topic outputs.

**What Bedrock Guardrails can block:**

| Category | Example |
|---|---|
| **Topic restrictions** | "Don't discuss competitor products" |
| **Harmful content** | Hate speech, violence, explicit content |
| **Sensitive information** | Block PII like credit card numbers |
| **Grounding** | Flag responses not based in fact |
| **Word filters** | Block specific words or phrases |

**The full Bedrock feature map:**

| Feature | Purpose |
|---|---|
| **Knowledge Bases** | Connect model to your documents (RAG) |
| **Agents** | Multi-step actions and API calls |
| **Fine-tuning** | Adapt model to your domain |
| **Guardrails** | Control what the model can say |


### Q37 — Semi-Supervised Learning

A medical imaging company has 50,000 X-ray scans. Only 2,000 of them have been labeled by doctors (expensive and time-consuming). A researcher proposes training the model using *both* the 2,000 labeled scans and the 48,000 unlabeled ones together. What type of ML is this?

- A) Supervised learning
- B) Unsupervised learning
- C) Reinforcement learning
- D) Semi-supervised learning

**✅ Answer: D — Semi-supervised learning**

A mix of a small amount of labeled data and a large amount of unlabeled data trained together. It's the practical middle ground when labeling data is expensive. Watch for phrases like *"only a small portion is labeled"* or *"labeling is expensive."*

---

### Q38 — Responsible AI: Explainability

A bank's AI model denies a customer's loan application. The customer asks why they were rejected, but the system only returns a score with no reasoning. A regulator flags this as a compliance concern. Which responsible AI principle is most directly violated?

- A) Fairness
- B) Robustness
- C) Explainability
- D) Privacy

**✅ Answer: C — Explainability**

Explainability is the principle that users (and regulators) should be able to understand *why* a model made a decision. A score with no reasoning is a classic black box problem. Note: fairness is about *who* is affected, explainability is about *why* a decision was made.

---

### Q39 — Context Window

A developer is using a foundation model on Amazon Bedrock to summarize legal contracts. She notices that when she submits very long contracts, the model seems to ignore or "forget" content from earlier in the document. What is the most likely cause?

- A) The model is hallucinating
- B) The input has exceeded the model's context window
- C) The model needs fine-tuning on legal documents
- D) Model drift has occurred

**✅ Answer: B — The input has exceeded the model's context window**

The context window is the maximum amount of text a model can "see" at one time — once exceeded, earlier content gets dropped or ignored. Watch for phrases like *"forgets earlier parts"*, *"long documents"*, or *"ignores context from earlier."*

---

### Q40 — AWS AI Services

A retail company wants to automatically extract the total amount, vendor name, and date from thousands of scanned paper invoices stored in S3 — without training a custom ML model. Which AWS service is best suited for this?

- A) Amazon Rekognition
- B) Amazon Comprehend
- C) Amazon Textract
- D) Amazon SageMaker

**✅ Answer: C — Amazon Textract**

| Service | What it does |
|---|---|
| **Textract** | Extracts text, forms, and structured data from scanned documents |
| **Rekognition** | Detects objects, faces, and labels in images |
| **Comprehend** | Understands and analyzes already extracted text (sentiment, entities) |
| **SageMaker** | Build, train, and deploy custom ML models |

The giveaways are *"scanned paper invoices"* and *"without training a custom model."*
