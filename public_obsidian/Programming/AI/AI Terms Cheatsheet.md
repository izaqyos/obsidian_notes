
- [1. Foundational Concepts](#1.%20Foundational%20Concepts)
- [2. Supervised Learning](#2.%20Supervised%20Learning)
- [3. Unsupervised Learning](#3.%20Unsupervised%20Learning)
- [4. Reinforcement Learning](#4.%20Reinforcement%20Learning)
- [5. Neural Networks Fundamentals](#5.%20Neural%20Networks%20Fundamentals)
- [6. Convolutional Neural Networks (CNNs)](#6.%20Convolutional%20Neural%20Networks%20(CNNs))
- [7. Recurrent Neural Networks (RNNs)](#7.%20Recurrent%20Neural%20Networks%20(RNNs))
- [8. Transformers & Attention](#8.%20Transformers%20&%20Attention)
- [9. Large Language Models (LLMs)](#9.%20Large%20Language%20Models%20(LLMs))
- [10. Generative AI](#10.%20Generative%20AI)
- [11. Model Training & Optimization](#11.%20Model%20Training%20&%20Optimization)
- [12. Model Evaluation & Metrics](#12.%20Model%20Evaluation%20&%20Metrics)
- [13. LLM Techniques & Alignment](#13.%20LLM%20Techniques%20&%20Alignment)
- [14. Efficient ML & Deployment](#14.%20Efficient%20ML%20&%20Deployment)
- [15. Multimodal AI](#15.%20Multimodal%20AI)
- [16. Data & Preprocessing](#16.%20Data%20&%20Preprocessing)
- [17. AI Safety & Ethics](#17.%20AI%20Safety%20&%20Ethics)
- [18. Infrastructure & Scaling](#18.%20Infrastructure%20&%20Scaling)
- [Quick Reference: Popular Models](#Quick%20Reference:%20Popular%20Models)
- [Overview](#Overview)
- [Key Concepts Explained](#Key%20Concepts%20Explained)
- [Learn More](#Learn%20More)
- [Overview](#Overview)
- [Types](#Types)
- [Algorithms Explained](#Algorithms%20Explained)
- [Learn More](#Learn%20More)
- [Overview](#Overview)
- [Types](#Types)
- [Algorithms Explained](#Algorithms%20Explained)
- [Learn More](#Learn%20More)
- [Overview](#Overview)
- [Key Components](#Key%20Components)
- [Algorithms](#Algorithms)
- [Learn More](#Learn%20More)
- [Overview](#Overview)
- [Architecture Components](#Architecture%20Components)
- [Training Process](#Training%20Process)
- [Key Concepts](#Key%20Concepts)
- [Learn More](#Learn%20More)
- [Overview](#Overview)
- [Key Operations](#Key%20Operations)
- [CNN Architecture](#CNN%20Architecture)
- [Famous CNN Architectures](#Famous%20CNN%20Architectures)
- [Learn More](#Learn%20More)
- [Overview](#Overview)
- [The Problem with Feedforward Networks](#The%20Problem%20with%20Feedforward%20Networks)
- [RNN Architecture](#RNN%20Architecture)
- [The Vanishing Gradient Problem](#The%20Vanishing%20Gradient%20Problem)
- [LSTM (Long Short-Term Memory)](#LSTM%20(Long%20Short-Term%20Memory))
- [GRU (Gated Recurrent Unit)](#GRU%20(Gated%20Recurrent%20Unit))
- [Bidirectional RNN](#Bidirectional%20RNN)
- [Learn More](#Learn%20More)
- [Overview](#Overview)
- [Why Attention?](#Why%20Attention?)
- [Self-Attention Mechanism](#Self-Attention%20Mechanism)
- [Transformer Architecture](#Transformer%20Architecture)
- [Architecture Variants](#Architecture%20Variants)
- [Positional Encoding](#Positional%20Encoding)
- [Learn More](#Learn%20More)
- [Overview](#Overview)
- [How LLMs Work](#How%20LLMs%20Work)
- [Key Concepts](#Key%20Concepts)
- [Training Pipeline](#Training%20Pipeline)
- [Prompting Techniques](#Prompting%20Techniques)
- [Learn More](#Learn%20More)
- [Overview](#Overview)
- [Generative vs Discriminative](#Generative%20vs%20Discriminative)
- [Main Approaches](#Main%20Approaches)
	- [1. GANs (Generative Adversarial Networks)](#1.%20GANs%20(Generative%20Adversarial%20Networks))

# AI & Machine Learning Terminology Cheat Sheet

---

## 1. Foundational Concepts

| Term | Definition |
|------|------------|
| **Artificial Intelligence (AI)** | Systems designed to perform tasks that typically require human intelligence |
| **Machine Learning (ML)** | Subset of AI where systems learn patterns from data without explicit programming |
| **Deep Learning (DL)** | ML using neural networks with multiple layers to learn hierarchical representations |
| **Model** | A mathematical representation learned from data to make predictions |
| **Training** | The process of teaching a model by exposing it to data |
| **Inference** | Using a trained model to make predictions on new data |
| **Feature** | An individual measurable property used as input to a model |
| **Label** | The target output a model learns to predict (in supervised learning) |
| **Dataset** | A collection of data used for training, validation, or testing |
| **Hyperparameter** | Configuration settings set before training (e.g., learning rate, batch size) |
| **Parameter** | Values the model learns during training (e.g., weights, biases) |

---

## 2. Supervised Learning

| Term | Definition |
|------|------------|
| **Supervised Learning** | Learning from labeled data with known input-output pairs |
| **Regression** | Predicting continuous numerical values (e.g., price, temperature) |
| **Classification** | Predicting discrete categories or classes |
| **Linear Regression** | Models linear relationship between features and continuous target |
| **Logistic Regression** | Classification algorithm using sigmoid function for probability output |
| **Decision Tree** | Tree-structured model making decisions based on feature thresholds |
| **Random Forest** | Ensemble of decision trees that vote on predictions |
| **Support Vector Machine (SVM)** | Finds optimal hyperplane to separate classes |
| **k-Nearest Neighbors (k-NN)** | Classifies based on majority vote of k closest training examples |
| **Naive Bayes** | Probabilistic classifier based on Bayes' theorem with feature independence assumption |
| **Gradient Boosting** | Ensemble method building models sequentially to correct errors (XGBoost, LightGBM) |

---

## 3. Unsupervised Learning

| Term | Definition |
|------|------------|
| **Unsupervised Learning** | Learning patterns from unlabeled data |
| **Clustering** | Grouping similar data points together |
| **K-Means** | Partitions data into k clusters based on centroid distance |
| **Hierarchical Clustering** | Creates tree of clusters (dendrogram) through merging or splitting |
| **DBSCAN** | Density-based clustering that finds arbitrarily shaped clusters |
| **Dimensionality Reduction** | Reducing number of features while preserving important information |
| **Principal Component Analysis (PCA)** | Linear technique projecting data onto principal components |
| **t-SNE** | Non-linear technique for visualizing high-dimensional data in 2D/3D |
| **UMAP** | Fast dimensionality reduction preserving both local and global structure |
| **Anomaly Detection** | Identifying unusual patterns or outliers in data |
| **Association Rules** | Discovering relationships between variables (e.g., market basket analysis) |

---

## 4. Reinforcement Learning

| Term | Definition |
|------|------------|
| **Reinforcement Learning (RL)** | Learning through interaction with environment via rewards and penalties |
| **Agent** | The learner or decision-maker |
| **Environment** | The world the agent interacts with |
| **State** | Current situation of the agent |
| **Action** | Choices available to the agent |
| **Reward** | Feedback signal indicating action quality |
| **Policy** | Strategy mapping states to actions |
| **Value Function** | Expected cumulative reward from a state |
| **Q-Learning** | Learns action-value function to find optimal policy |
| **Deep Q-Network (DQN)** | Q-learning with deep neural networks |
| **Policy Gradient** | Directly optimizes the policy using gradient ascent |
| **Actor-Critic** | Combines policy-based (actor) and value-based (critic) methods |
| **RLHF** | Reinforcement Learning from Human Feedback—training with human preferences |
| **Exploration vs Exploitation** | Balancing trying new actions vs using known good actions |

---

## 5. Neural Networks Fundamentals

| Term | Definition |
|------|------------|
| **Neural Network** | Computing system inspired by biological neural networks |
| **Neuron/Node** | Basic unit that receives inputs, applies weights, and outputs activation |
| **Layer** | Collection of neurons at the same depth in the network |
| **Input Layer** | First layer receiving raw data |
| **Hidden Layer** | Intermediate layers between input and output |
| **Output Layer** | Final layer producing predictions |
| **Weights** | Learnable parameters determining connection strength |
| **Bias** | Learnable offset added to weighted sum |
| **Activation Function** | Non-linear function applied to neuron output (ReLU, Sigmoid, Tanh) |
| **Feedforward** | Information flows in one direction from input to output |
| **Backpropagation** | Algorithm computing gradients by propagating errors backward |
| **Gradient Descent** | Optimization algorithm minimizing loss by following gradient |
| **Learning Rate** | Step size for weight updates during training |
| **Batch Size** | Number of samples processed before updating weights |
| **Epoch** | One complete pass through the entire training dataset |

---

## 6. Convolutional Neural Networks (CNNs)

| Term | Definition |
|------|------------|
| **CNN** | Neural network specialized for grid-like data (images, time series) |
| **Convolution** | Operation applying filters to detect local patterns |
| **Filter/Kernel** | Small matrix of weights that slides across input |
| **Feature Map** | Output of applying a filter to input |
| **Stride** | Step size of filter movement |
| **Padding** | Adding borders to input to control output size |
| **Pooling** | Downsampling operation reducing spatial dimensions |
| **Max Pooling** | Takes maximum value in each pooling window |
| **Average Pooling** | Takes average value in each pooling window |
| **Receptive Field** | Region of input that influences a particular neuron |
| **ResNet** | Architecture using skip connections to train very deep networks |
| **VGG** | Deep CNN architecture with small 3×3 filters |

---

## 7. Recurrent Neural Networks (RNNs)

| Term | Definition |
|------|------------|
| **RNN** | Neural network with connections forming cycles for sequential data |
| **Hidden State** | Memory carrying information across time steps |
| **Vanishing Gradient** | Gradients shrinking to zero in long sequences, preventing learning |
| **Exploding Gradient** | Gradients growing unboundedly, causing unstable training |
| **LSTM** | Long Short-Term Memory—RNN variant with gates controlling information flow |
| **GRU** | Gated Recurrent Unit—simplified LSTM with fewer parameters |
| **Bidirectional RNN** | Processes sequence in both forward and backward directions |
| **Sequence-to-Sequence** | Architecture mapping input sequence to output sequence |
| **Teacher Forcing** | Training technique using ground truth as next input |

---

## 8. Transformers & Attention

| Term | Definition |
|------|------------|
| **Transformer** | Architecture using self-attention, replacing recurrence entirely |
| **Attention** | Mechanism allowing model to focus on relevant parts of input |
| **Self-Attention** | Attention where query, key, and value come from same sequence |
| **Multi-Head Attention** | Multiple attention mechanisms running in parallel |
| **Query, Key, Value (Q, K, V)** | Vectors used to compute attention weights |
| **Positional Encoding** | Information added to embeddings to convey sequence position |
| **Encoder** | Processes input sequence into contextual representations |
| **Decoder** | Generates output sequence from encoder representations |
| **Encoder-Only** | Architecture for understanding tasks (e.g., BERT) |
| **Decoder-Only** | Architecture for generation tasks (e.g., GPT) |
| **Causal/Masked Attention** | Prevents attending to future tokens during generation |
| **Cross-Attention** | Decoder attending to encoder outputs |
| **Layer Normalization** | Normalizes activations across features for stable training |
| **Feed-Forward Network (FFN)** | Dense layers applied position-wise in transformers |

---

## 9. Large Language Models (LLMs)

| Term | Definition |
|------|------------|
| **LLM** | Large-scale neural network trained on vast text data for language tasks |
| **Pre-training** | Initial training on large unlabeled corpus (e.g., next-token prediction) |
| **Fine-tuning** | Adapting pre-trained model to specific task with labeled data |
| **Transfer Learning** | Applying knowledge from one task/domain to another |
| **Zero-Shot** | Performing task without any task-specific examples |
| **Few-Shot** | Performing task with only a few examples provided |
| **In-Context Learning** | Learning from examples provided in the prompt |
| **Prompt** | Input text given to guide model output |
| **Prompt Engineering** | Crafting prompts to elicit desired model behavior |
| **Tokenization** | Splitting text into tokens (subwords, words, or characters) |
| **Vocabulary** | Set of all tokens the model recognizes |
| **Embedding** | Dense vector representation of tokens or concepts |
| **Context Window** | Maximum number of tokens model can process at once |
| **Autoregressive** | Generating output one token at a time, conditioned on previous tokens |

---

## 10. Generative AI

| Term | Definition |
|------|------------|
| **Generative AI** | AI systems that create new content (text, images, audio, video) |
| **Generative Model** | Model learning data distribution to generate new samples |
| **Discriminative Model** | Model learning decision boundary between classes |
| **GAN** | Generative Adversarial Network—generator and discriminator competing |
| **Generator** | Network producing synthetic data |
| **Discriminator** | Network distinguishing real from generated data |
| **VAE** | Variational Autoencoder—learns latent space for generation |
| **Latent Space** | Compressed representation capturing data's underlying structure |
| **Diffusion Model** | Generates data by learning to reverse a noise-adding process |
| **Denoising** | Removing noise to recover original signal |
| **Stable Diffusion** | Open-source diffusion model for image generation |
| **DALL-E** | OpenAI's text-to-image generation model |
| **Midjourney** | AI image generation service |
| **Text-to-Image** | Generating images from text descriptions |
| **Image-to-Image** | Transforming one image into another |
| **Inpainting** | Filling in missing or masked regions of an image |
| **Style Transfer** | Applying artistic style of one image to content of another |

---

## 11. Model Training & Optimization

| Term | Definition |
|------|------------|
| **Loss Function** | Measures difference between predictions and actual values |
| **Cross-Entropy Loss** | Common loss for classification tasks |
| **Mean Squared Error (MSE)** | Common loss for regression tasks |
| **Optimizer** | Algorithm updating weights to minimize loss |
| **SGD** | Stochastic Gradient Descent—updates using random data subsets |
| **Adam** | Adaptive optimizer combining momentum and RMSprop |
| **Momentum** | Accelerates gradient descent by accumulating velocity |
| **Learning Rate Scheduler** | Adjusts learning rate during training |
| **Warmup** | Gradually increasing learning rate at training start |
| **Gradient Clipping** | Limiting gradient magnitude to prevent explosion |
| **Regularization** | Techniques preventing overfitting (L1, L2, dropout) |
| **Dropout** | Randomly deactivating neurons during training |
| **Batch Normalization** | Normalizing layer inputs to accelerate training |
| **Early Stopping** | Halting training when validation performance degrades |
| **Checkpointing** | Saving model state during training |

---

## 12. Model Evaluation & Metrics

| Term | Definition |
|------|------------|
| **Training Set** | Data used to train the model |
| **Validation Set** | Data used to tune hyperparameters and monitor training |
| **Test Set** | Held-out data for final performance evaluation |
| **Overfitting** | Model memorizes training data, performs poorly on new data |
| **Underfitting** | Model too simple to capture underlying patterns |
| **Bias-Variance Tradeoff** | Balance between model simplicity and flexibility |
| **Cross-Validation** | Evaluating model on multiple data splits |
| **Accuracy** | Proportion of correct predictions |
| **Precision** | Proportion of positive predictions that are correct |
| **Recall/Sensitivity** | Proportion of actual positives correctly identified |
| **F1 Score** | Harmonic mean of precision and recall |
| **AUC-ROC** | Area under receiver operating characteristic curve |
| **Confusion Matrix** | Table showing true/false positives and negatives |
| **Perplexity** | Measures how well model predicts a sample (lower is better) |
| **BLEU** | Evaluates text generation quality against references |
| **ROUGE** | Evaluates summarization by comparing n-gram overlap |

---

## 13. LLM Techniques & Alignment

| Term | Definition |
|------|------------|
| **Instruction Tuning** | Fine-tuning on instruction-response pairs |
| **Constitutional AI** | Training AI to follow principles through self-critique |
| **RLHF** | Reinforcement Learning from Human Feedback |
| **Reward Model** | Model predicting human preferences for RLHF |
| **DPO** | Direct Preference Optimization—simpler alternative to RLHF |
| **Alignment** | Ensuring AI behaves according to human values and intentions |
| **Hallucination** | Model generating plausible but factually incorrect information |
| **Grounding** | Connecting model outputs to verifiable sources |
| **Chain-of-Thought (CoT)** | Prompting model to show reasoning steps |
| **Retrieval-Augmented Generation (RAG)** | Combining retrieval with generation for accuracy |
| **Tool Use/Function Calling** | LLM invoking external tools or APIs |
| **Agents** | AI systems that plan and execute multi-step tasks |
| **Guardrails** | Safety mechanisms constraining model outputs |

---

## 14. Efficient ML & Deployment

| Term | Definition |
|------|------------|
| **Quantization** | Reducing numerical precision to shrink model size |
| **Pruning** | Removing unimportant weights or neurons |
| **Knowledge Distillation** | Training smaller model to mimic larger one |
| **LoRA** | Low-Rank Adaptation—efficient fine-tuning by training small adapters |
| **QLoRA** | Quantized LoRA—combining quantization with LoRA |
| **PEFT** | Parameter-Efficient Fine-Tuning methods |
| **Model Compression** | Techniques reducing model size and computational cost |
| **Inference Optimization** | Speeding up model predictions |
| **Batching** | Processing multiple inputs simultaneously |
| **Caching** | Storing computed values (e.g., KV cache in transformers) |
| **Speculative Decoding** | Using draft model to accelerate generation |
| **Edge Deployment** | Running models on devices with limited resources |
| **Model Serving** | Infrastructure for deploying models in production |
| **Latency** | Time delay from input to output |
| **Throughput** | Number of inferences processed per time unit |

---

## 15. Multimodal AI

| Term | Definition |
|------|------------|
| **Multimodal** | AI processing multiple data types (text, images, audio, video) |
| **Vision-Language Model (VLM)** | Model understanding both images and text |
| **CLIP** | Contrastive Language-Image Pre-training by OpenAI |
| **Image Encoder** | Converts images to embeddings |
| **Cross-Modal Attention** | Attention between different modalities |
| **Image Captioning** | Generating text descriptions of images |
| **Visual Question Answering (VQA)** | Answering questions about images |
| **OCR** | Optical Character Recognition—extracting text from images |
| **Speech-to-Text (STT)** | Converting spoken audio to text |
| **Text-to-Speech (TTS)** | Generating spoken audio from text |
| **Whisper** | OpenAI's speech recognition model |

---

## 16. Data & Preprocessing

| Term | Definition |
|------|------------|
| **Data Augmentation** | Creating variations of training data to improve generalization |
| **Normalization** | Scaling features to standard range |
| **Standardization** | Transforming features to zero mean and unit variance |
| **One-Hot Encoding** | Converting categories to binary vectors |
| **Word Embedding** | Dense vector representation of words (Word2Vec, GloVe) |
| **Subword Tokenization** | Breaking words into smaller units (BPE, WordPiece) |
| **BPE** | Byte Pair Encoding—common subword tokenization method |
| **Padding** | Adding tokens to make sequences equal length |
| **Masking** | Hiding certain inputs for training objectives |
| **Data Leakage** | Training data contaminating test evaluation |
| **Class Imbalance** | Unequal distribution of classes in dataset |
| **Synthetic Data** | Artificially generated data for training |

---

## 17. AI Safety & Ethics

| Term | Definition |
|------|------------|
| **AI Safety** | Research ensuring AI systems are beneficial and controllable |
| **Interpretability** | Understanding how models make decisions |
| **Explainability** | Providing human-understandable explanations for outputs |
| **Bias** | Systematic unfairness in model predictions |
| **Fairness** | Ensuring equitable treatment across groups |
| **Red Teaming** | Adversarial testing to find model vulnerabilities |
| **Jailbreaking** | Attempts to bypass model safety constraints |
| **Prompt Injection** | Malicious inputs manipulating model behavior |
| **Adversarial Examples** | Inputs designed to fool models |
| **Robustness** | Model resilience to perturbations and attacks |
| **Watermarking** | Embedding identifiable patterns in generated content |
| **Model Cards** | Documentation of model capabilities, limitations, and biases |

---

## 18. Infrastructure & Scaling

| Term | Definition |
|------|------------|
| **GPU** | Graphics Processing Unit—hardware accelerating parallel computation |
| **TPU** | Tensor Processing Unit—Google's custom ML accelerator |
| **CUDA** | NVIDIA's parallel computing platform |
| **Distributed Training** | Training across multiple devices or machines |
| **Data Parallelism** | Splitting data batches across devices |
| **Model Parallelism** | Splitting model layers across devices |
| **Pipeline Parallelism** | Different stages of model on different devices |
| **Mixed Precision** | Using lower precision (FP16) for speed with FP32 for stability |
| **Tensor** | Multi-dimensional array, fundamental data structure in ML |
| **FLOPS** | Floating Point Operations Per Second—computational metric |
| **Scaling Laws** | Relationships between model size, data, compute, and performance |
| **Chinchilla Scaling** | Optimal ratio of parameters to training tokens |

---

## Quick Reference: Popular Models

| Category | Models |
|----------|--------|
| **LLMs** | GPT-4, Claude, Gemini, Llama, Mistral, Command R |
| **Image Generation** | DALL-E 3, Midjourney, Stable Diffusion, Imagen |
| **Vision-Language** | GPT-4V, Claude 3, Gemini, LLaVA |
| **Embeddings** | OpenAI Ada, Cohere Embed, BGE, E5 |
| **Speech** | Whisper, Eleven Labs, VALL-E |
| **Open Source LLMs** | Llama 3, Mistral, Falcon, Phi, Qwen |

# AI & Machine Learning Comprehensive Guide

---

# 1. Foundational Concepts

## Overview
Machine Learning is the science of getting computers to learn from data without being explicitly programmed. Instead of writing rules, we show the system examples and let it discover patterns.

## Key Concepts Explained

**Artificial Intelligence (AI)** — The broad field of creating systems that can perform tasks requiring human-like intelligence: reasoning, learning, perception, and decision-making.

**Machine Learning (ML)** — A subset of AI where algorithms improve through experience. The system learns patterns from data rather than following hard-coded rules.

**Deep Learning (DL)** — ML using neural networks with many layers. "Deep" refers to the number of layers. These networks automatically learn hierarchical features (e.g., edges → shapes → objects in images).

**Model** — The mathematical function learned from data. Think of it as a sophisticated equation with millions of adjustable parameters that maps inputs to outputs.

**Training vs Inference**
- *Training*: The learning phase where the model adjusts its parameters using data
- *Inference*: Using the trained model to make predictions on new, unseen data

**Features vs Labels**
- *Features*: Input variables (what the model sees) — e.g., pixel values, word frequencies
- *Labels*: Target outputs (what the model predicts) — e.g., "cat" or "dog"

**Parameters vs Hyperparameters**
- *Parameters*: Learned by the model (weights, biases) — millions or billions of values
- *Hyperparameters*: Set by humans before training (learning rate, layers, batch size)

```
┌─────────────────────────────────────────────────────────────┐
│                    MACHINE LEARNING PIPELINE                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────────────┐     │
│   │   DATA   │───▶│  MODEL   │───▶│    PREDICTIONS   │     │
│   │ (inputs) │    │(learning)│    │    (outputs)     │     │
│   └──────────┘    └────┬─────┘    └──────────────────┘     │
│                        │                    │               │
│                        │    ┌───────────────┘               │
│                        ▼    ▼                               │
│                   ┌──────────────┐                          │
│                   │ LOSS/ERROR   │◀── Compare with          │
│                   │  FUNCTION    │    actual labels         │
│                   └──────┬───────┘                          │
│                          │                                  │
│                          ▼                                  │
│                   ┌──────────────┐                          │
│                   │   UPDATE     │                          │
│                   │  PARAMETERS  │──── Repeat until         │
│                   └──────────────┘     good enough          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Learn More
- **Course**: [Andrew Ng's Machine Learning (Coursera)](https://www.coursera.org/learn/machine-learning)
- **Book**: "Hands-On Machine Learning" by Aurélien Géron
- **Interactive**: [Google's ML Crash Course](https://developers.google.com/machine-learning/crash-course)
- **Visual**: [3Blue1Brown Neural Networks Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

---

# 2. Supervised Learning

## Overview
Supervised learning uses labeled data — examples where we know the correct answer. The model learns to map inputs to outputs by studying these examples, then generalizes to new data.

## Types

**Classification** — Predicting discrete categories
- Binary: Yes/No, Spam/Not Spam
- Multi-class: Cat/Dog/Bird
- Multi-label: An image can be "outdoor," "sunny," AND "beach"

**Regression** — Predicting continuous values
- House prices, temperature, stock prices

## Algorithms Explained

| Algorithm | How It Works | Best For | Limitations |
|-----------|--------------|----------|-------------|
| **Linear Regression** | Fits a straight line through data points minimizing squared errors | Simple relationships, baseline models | Can't capture non-linear patterns |
| **Logistic Regression** | Uses sigmoid function to output probabilities (0-1) for classification | Binary classification, interpretable results | Linear decision boundaries only |
| **Decision Tree** | Creates if-then rules by splitting data on feature thresholds | Interpretable models, mixed data types | Prone to overfitting |
| **Random Forest** | Ensemble of many decision trees that vote; reduces overfitting | General-purpose, handles noise well | Less interpretable, slower |
| **Gradient Boosting** | Builds trees sequentially, each correcting previous errors | Competitions, tabular data (XGBoost, LightGBM) | Can overfit, requires tuning |
| **SVM** | Finds hyperplane maximizing margin between classes | High-dimensional data, clear margins | Slow on large datasets |
| **k-NN** | Classifies based on k nearest neighbors' majority vote | Simple problems, no training needed | Slow at inference, curse of dimensionality |
| **Naive Bayes** | Applies Bayes theorem assuming feature independence | Text classification, fast training | Independence assumption often wrong |

```
┌─────────────────────────────────────────────────────────────┐
│                   SUPERVISED LEARNING FLOW                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   TRAINING DATA (Labeled)                                   │
│   ┌─────────────────────────────────────────┐               │
│   │ Features (X)          │ Label (Y)       │               │
│   ├───────────────────────┼─────────────────┤               │
│   │ [5.1, 3.5, 1.4, 0.2]  │ Setosa          │               │
│   │ [7.0, 3.2, 4.7, 1.4]  │ Versicolor      │               │
│   │ [6.3, 3.3, 6.0, 2.5]  │ Virginica       │               │
│   └───────────────────────┴─────────────────┘               │
│                      │                                      │
│                      ▼                                      │
│              ┌───────────────┐                              │
│              │   TRAINING    │                              │
│              │   ALGORITHM   │                              │
│              └───────┬───────┘                              │
│                      │                                      │
│                      ▼                                      │
│              ┌───────────────┐                              │
│              │ TRAINED MODEL │                              │
│              └───────┬───────┘                              │
│                      │                                      │
│   NEW DATA           ▼                                      │
│   [6.5, 3.0, 5.2, 2.0] ──▶ Model ──▶ "Virginica" (93%)     │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              CLASSIFICATION VS REGRESSION                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   CLASSIFICATION                  REGRESSION                │
│   (Discrete Output)               (Continuous Output)       │
│                                                             │
│       ●  ●                            ●                     │
│     ●  ●  ●     Class A                  ●    ●             │
│       ●  ●                            ●    ●                │
│   ─────────────────              ────●────────────          │
│       ○  ○                        ●     Fitted              │
│     ○  ○  ○     Class B            ●      Line              │
│       ○  ○                      ●                           │
│                                                             │
│   Output: "Class A" or "B"       Output: 23.7, 156.2, etc  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Learn More
- **Course**: [Stanford CS229 (YouTube)](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)
- **Practice**: [Kaggle Learn](https://www.kaggle.com/learn)
- **Book**: "An Introduction to Statistical Learning" (free PDF)
- **Interactive**: [Scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html)

---

# 3. Unsupervised Learning

## Overview
Unsupervised learning finds hidden patterns in data without labels. The algorithm discovers structure on its own — grouping similar items, reducing complexity, or finding anomalies.

## Types

**Clustering** — Grouping similar data points
- Customer segmentation, document organization, image grouping

**Dimensionality Reduction** — Compressing data while preserving information
- Visualization, noise reduction, feature extraction

**Anomaly Detection** — Finding unusual patterns
- Fraud detection, system monitoring, quality control

## Algorithms Explained

| Algorithm | How It Works | Best For |
|-----------|--------------|----------|
| **K-Means** | Iteratively assigns points to k cluster centroids, updates centroids | Fast clustering when k is known |
| **Hierarchical** | Builds tree of clusters by merging (agglomerative) or splitting (divisive) | When cluster hierarchy matters |
| **DBSCAN** | Groups points in dense regions, marks sparse points as outliers | Arbitrary shapes, outlier detection |
| **PCA** | Projects data onto directions of maximum variance (principal components) | Linear dimensionality reduction |
| **t-SNE** | Preserves local neighborhoods in low-dimensional embedding | Visualization of clusters |
| **UMAP** | Faster than t-SNE, preserves more global structure | Large-scale visualization |
| **Autoencoders** | Neural network learns to compress and reconstruct data | Non-linear reduction, denoising |

```
┌─────────────────────────────────────────────────────────────┐
│                    CLUSTERING ALGORITHMS                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   K-MEANS                         DBSCAN                    │
│   (Spherical clusters)            (Density-based)           │
│                                                             │
│       ┌───●─●─┐                    ●●●●                     │
│       │ ● ✕ ● │ Cluster 1           ●●●●    Cluster 1      │
│       │  ●●   │                      ●●                     │
│       └───────┘                                             │
│                                        ○ ← Outlier/Noise    │
│       ┌───●─●─┐                                             │
│       │ ● ✕ ● │ Cluster 2         ●●●●●●●                   │
│       │  ●●   │                    ●●●●●●  Cluster 2        │
│       └───────┘                     ●●●●                    │
│                                                             │
│   ✕ = Centroid                    Finds arbitrary shapes    │
│   Assumes circular clusters       Handles outliers well     │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                 DIMENSIONALITY REDUCTION                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   HIGH DIMENSIONAL                 LOW DIMENSIONAL          │
│   (Hard to visualize)              (2D/3D visualization)    │
│                                                             │
│   Features: [x1, x2, x3,           ┌─────────────────┐      │
│              x4, x5, ...,    ───▶  │    ●  ●         │      │
│              x100]                 │  ●  ●  ●        │      │
│                                    │    ●    ●  ●    │      │
│   100 dimensions                   │  ○  ○           │      │
│   can't visualize                  │    ○  ○  ○      │      │
│                                    └─────────────────┘      │
│                                    2D: Clusters visible!    │
│                                                             │
│   Methods: PCA (linear), t-SNE/UMAP (non-linear)           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Learn More
- **Course**: [Unsupervised Learning (Coursera)](https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning)
- **Visual**: [Visualizing K-Means](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)
- **Tool**: [Embedding Projector (TensorFlow)](https://projector.tensorflow.org/)
- **Paper**: [UMAP Paper](https://arxiv.org/abs/1802.03426)

---

# 4. Reinforcement Learning

## Overview
Reinforcement Learning (RL) trains agents to make decisions through trial and error. The agent interacts with an environment, receives rewards or penalties, and learns a policy to maximize cumulative reward.

## Key Components

| Component | Description | Example (Game) |
|-----------|-------------|----------------|
| **Agent** | The learner/decision-maker | The AI player |
| **Environment** | The world the agent operates in | The game world |
| **State (s)** | Current situation | Player position, score, enemies |
| **Action (a)** | Choices available | Move left, jump, shoot |
| **Reward (r)** | Feedback signal | +10 for coin, -100 for death |
| **Policy (π)** | Strategy: state → action | "If enemy near, jump" |
| **Value Function** | Expected future reward from state | "This position is worth 50 points" |

## Algorithms

| Algorithm | Type | Key Idea |
|-----------|------|----------|
| **Q-Learning** | Value-based | Learns Q(s,a) = expected reward for action a in state s |
| **DQN** | Deep Value-based | Q-learning with neural networks (Atari games) |
| **Policy Gradient** | Policy-based | Directly optimizes the policy using gradients |
| **A2C/A3C** | Actor-Critic | Actor (policy) + Critic (value) working together |
| **PPO** | Policy-based | Stable policy updates with clipping (widely used) |
| **SAC** | Actor-Critic | Maximum entropy for better exploration |
| **RLHF** | Alignment | Uses human preferences as reward signal (ChatGPT) |

```
┌─────────────────────────────────────────────────────────────┐
│                  REINFORCEMENT LEARNING LOOP                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    ┌─────────────────┐                      │
│         State(t)   │                 │   Action(t)          │
│      ┌────────────▶│      AGENT      │◀────────────┐        │
│      │             │    (Policy π)   │             │        │
│      │             └────────┬────────┘             │        │
│      │                      │                      │        │
│      │                      │ Action(t)            │        │
│      │                      ▼                      │        │
│      │             ┌─────────────────┐             │        │
│      │             │                 │             │        │
│      └─────────────│   ENVIRONMENT   │─────────────┘        │
│        State(t+1)  │                 │   Reward(t)          │
│                    └─────────────────┘                      │
│                                                             │
│   Goal: Learn policy π that maximizes cumulative reward     │
│         Σ γᵗ × r(t)  where γ is discount factor            │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              EXPLORATION VS EXPLOITATION                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   EXPLOITATION                    EXPLORATION               │
│   (Use what you know)             (Try new things)          │
│                                                             │
│   "I know this restaurant         "Maybe that new place     │
│    is good, let's go there"        is even better?"         │
│                                                             │
│          ┌───┐                         ┌───┐                │
│          │ A │ ← Known good            │ ? │ ← Unknown      │
│          │+10│                         │???│                │
│          └───┘                         └───┘                │
│                                                             │
│   ε-greedy: With probability ε, explore randomly            │
│             With probability 1-ε, exploit best known action │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                          RLHF                               │
│           (Reinforcement Learning from Human Feedback)      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Step 1: Pre-train LLM on text data                        │
│                                                             │
│   Step 2: Collect human preferences                         │
│           ┌────────────────────────────────────┐            │
│           │ Prompt: "Explain quantum physics"  │            │
│           │                                    │            │
│           │ Response A: [Technical jargon...]  │            │
│           │ Response B: [Clear explanation...] │            │
│           │                                    │            │
│           │ Human: "B is better" ✓             │            │
│           └────────────────────────────────────┘            │
│                                                             │
│   Step 3: Train Reward Model on preferences                 │
│           Reward Model learns: B > A                        │
│                                                             │
│   Step 4: Fine-tune LLM with RL (PPO)                       │
│           LLM generates → Reward Model scores → Update LLM  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Learn More
- **Course**: [Deep RL (Berkeley CS285)](https://rail.eecs.berkeley.edu/deeprlcourse/)
- **Book**: "Reinforcement Learning" by Sutton & Barto (free online)
- **Practice**: [OpenAI Gymnasium](https://gymnasium.farama.org/)
- **Visual**: [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/)

---

# 5. Neural Networks Fundamentals

## Overview
Neural networks are computing systems loosely inspired by biological brains. They consist of layers of interconnected nodes (neurons) that learn to transform inputs into outputs through training.

## Architecture Components

**Neuron/Node** — Receives inputs, applies weights and bias, passes through activation function

```
Inputs      Weights
  x₁ ───w₁───┐
             │
  x₂ ───w₂───┼──▶ Σ(wᵢxᵢ) + b ──▶ f(·) ──▶ Output
             │        ↑            ↑
  x₃ ───w₃───┘      Bias      Activation
                              Function
```

**Layers**
- *Input Layer*: Receives raw data (pixels, words, numbers)
- *Hidden Layers*: Learn intermediate representations
- *Output Layer*: Produces final prediction

**Activation Functions** — Add non-linearity (without them, network = linear function)

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| **ReLU** | max(0, x) | [0, ∞) | Hidden layers (most common) |
| **Sigmoid** | 1/(1+e⁻ˣ) | (0, 1) | Binary output, gates |
| **Tanh** | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | (-1, 1) | Hidden layers, centered output |
| **Softmax** | eˣⁱ/Σeˣʲ | (0, 1), sums to 1 | Multi-class output (probabilities) |

## Training Process

**Forward Pass**: Input flows through network to produce prediction
**Loss Calculation**: Compare prediction to actual label
**Backpropagation**: Compute gradients of loss w.r.t. each weight
**Weight Update**: Adjust weights to reduce loss (gradient descent)

```
┌─────────────────────────────────────────────────────────────┐
│                  NEURAL NETWORK ARCHITECTURE                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   INPUT        HIDDEN LAYER 1    HIDDEN LAYER 2   OUTPUT   │
│   LAYER                                                     │
│                                                             │
│     ○─────────────○                                         │
│      \           / \               ○                        │
│       \         /   \             /│\              ○        │
│     ○──\───────/─────○───────────/─┼─\────────────/         │
│         \     /     / \         /  │  \          /          │
│          \   /     /   \       /   │   \        /           │
│     ○─────\ /─────○─────\─────/────┼────○──────○            │
│            ╳             \   /     │    │                   │
│           / \             \ /      │    │                   │
│     ○────/───\────○────────╳───────┼────○                   │
│         /     \          / \       │                        │
│        /       \        /   \      │                        │
│     ○─/─────────\──────○─────\─────○                        │
│                  \            \                             │
│                   ○            ○                            │
│                                                             │
│   Features      Learning        Learning      Prediction    │
│   (e.g.,        low-level       high-level    (e.g.,        │
│   pixels)       patterns        concepts      "cat")        │
│                 (edges)         (shapes)                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                BACKPROPAGATION & GRADIENT DESCENT           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   1. FORWARD PASS                                           │
│      Input ──▶ Layer 1 ──▶ Layer 2 ──▶ Output ──▶ Loss     │
│                                                             │
│   2. BACKWARD PASS (Backpropagation)                        │
│      ∂Loss/∂w ◀── Layer 2 ◀── Layer 1 ◀── Output ◀── Loss  │
│      (Compute gradients using chain rule)                   │
│                                                             │
│   3. UPDATE WEIGHTS (Gradient Descent)                      │
│      w_new = w_old - learning_rate × ∂Loss/∂w               │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                    Loss Landscape                   │   │
│   │                         ╱╲                          │   │
│   │                        ╱  ╲                         │   │
│   │           ╱╲          ╱    ╲                        │   │
│   │          ╱  ╲        ╱      ╲     Current           │   │
│   │         ╱    ╲      ╱        ╲    position          │   │
│   │        ╱      ╲    ╱          ╲     ●               │   │
│   │       ╱        ╲  ╱            ╲   ↓                │   │
│   │      ╱          ╲╱              ╲  ↓  Follow        │   │
│   │     ╱       Minimum ●            ╲ ↓  gradient      │   │
│   │    ╱            ↑                 ╲↓  down          │   │
│   │   ╱          Goal!                 ●                │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   ACTIVATION FUNCTIONS                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ReLU                    Sigmoid                Tanh       │
│   f(x) = max(0,x)         f(x) = 1/(1+e⁻ˣ)      f(x)=tanh(x)│
│                                                             │
│        │    ╱                  ────────              ─────  │
│        │   ╱              ────╱                 ────╱       │
│        │  ╱              ╱                     ╱            │
│   ─────┼─╱────       ───╱──────────       ────╱─────        │
│        │╱               ╱                    ╱              │
│        │               ╱                    ╱               │
│        │              ╱                 ───╱                │
│                                        ─────                │
│                                                             │
│   Range: [0,∞)        Range: (0,1)         Range: (-1,1)   │
│   Fast, sparse        Probabilities        Zero-centered    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Concepts

| Term | Definition |
|------|------------|
| **Batch Size** | Number of samples processed before weight update (32, 64, 128 typical) |
| **Epoch** | One complete pass through training data |
| **Learning Rate** | Step size for weight updates (too high = unstable, too low = slow) |
| **Overfitting** | Model memorizes training data, fails on new data |
| **Regularization** | Techniques to prevent overfitting (dropout, L2, early stopping) |

## Learn More
- **Visual**: [3Blue1Brown Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- **Interactive**: [TensorFlow Playground](https://playground.tensorflow.org/)
- **Course**: [Fast.ai Practical Deep Learning](https://course.fast.ai/)
- **Book**: "Deep Learning" by Goodfellow, Bengio, Courville (free online)

---

# 6. Convolutional Neural Networks (CNNs)

## Overview
CNNs are specialized neural networks for processing grid-like data (images, audio spectrograms). They use convolutional layers that apply learnable filters to detect local patterns, achieving translation invariance.

## Key Operations

**Convolution** — Sliding a small filter across the image, computing dot products

```
Input Image (5×5)          Filter (3×3)         Output Feature Map
┌───┬───┬───┬───┬───┐     ┌───┬───┬───┐        ┌───┬───┬───┐
│ 1 │ 0 │ 1 │ 0 │ 1 │     │ 1 │ 0 │ 1 │        │   │   │   │
├───┼───┼───┼───┼───┤     ├───┼───┼───┤        ├───┼───┼───┤
│ 0 │ 1 │ 0 │ 1 │ 0 │  ✕  │ 0 │ 1 │ 0 │   =    │   │ 4 │   │
├───┼───┼───┼───┼───┤     ├───┼───┼───┤        ├───┼───┼───┤
│ 1 │ 0 │ 1 │ 0 │ 1 │     │ 1 │ 0 │ 1 │        │   │   │   │
├───┼───┼───┼───┼───┤     └───┴───┴───┘        └───┴───┴───┘
│ 0 │ 1 │ 0 │ 1 │ 0 │
├───┼───┼───┼───┼───┤     Filter slides across image
│ 1 │ 0 │ 1 │ 0 │ 1 │     detecting specific patterns
└───┴───┴───┴───┴───┘
```

**Pooling** — Downsampling to reduce spatial dimensions

| Type | Operation | Purpose |
|------|-----------|---------|
| **Max Pooling** | Takes maximum value in window | Preserves strongest activations |
| **Average Pooling** | Takes average value in window | Smooths features |
| **Global Pooling** | Pools entire feature map to single value | Reduces to fixed size |

## CNN Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CNN ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   INPUT        CONV      POOL     CONV      POOL     FC    │
│   IMAGE        LAYER     LAYER    LAYER     LAYER   LAYERS │
│                                                             │
│  ┌────────┐  ┌────────┐ ┌─────┐ ┌────────┐ ┌─────┐ ┌─────┐ │
│  │        │  │ ░░░░░░ │ │░░░░ │ │ ▓▓▓▓▓▓ │ │▓▓▓▓ │ │     │ │
│  │  🐱    │─▶│ ░░░░░░ │─▶░░░░ │─▶│ ▓▓▓▓▓▓ │─▶▓▓▓▓ │─▶ ○○○ │─▶ CAT
│  │        │  │ ░░░░░░ │ │░░░░ │ │ ▓▓▓▓▓▓ │ │▓▓▓▓ │ │ ○○○ │ │
│  └────────┘  └────────┘ └─────┘ └────────┘ └─────┘ └─────┘ │
│                                                             │
│  224×224×3   Multiple   Reduce   More      Reduce  Flatten │
│  (RGB)       filters    size     filters   size    + Dense │
│                                                             │
│  WHAT EACH LAYER LEARNS:                                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Layer 1  │ │ Layer 2  │ │ Layer 3  │ │ Layer 4+ │       │
│  │  Edges   │ │ Textures │ │  Parts   │ │ Objects  │       │
│  │ ╱ ╲ │ ─  │ │ ░░ ▓▓   │ │  👁 👃   │ │   🐱     │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  CONVOLUTION OPERATION                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Stride = 1 (move 1 pixel)      Stride = 2 (move 2 pixels)│
│   ┌─┬─┬─┬─┬─┐                    ┌─┬─┬─┬─┬─┐               │
│   │█│█│█│ │ │  →                 │█│█│█│ │ │  → →          │
│   │█│█│█│ │ │                    │█│█│█│ │ │               │
│   │█│█│█│ │ │                    │█│█│█│ │ │               │
│   │ │ │ │ │ │                    │ │ │ │ │ │               │
│   └─┴─┴─┴─┴─┘                    └─┴─┴─┴─┴─┘               │
│   Output: 3×3                    Output: 2×2               │
│                                                             │
│   Padding: Add zeros around border to preserve size         │
│   ┌─┬─┬─┬─┬─┬─┬─┐                                          │
│   │0│0│0│0│0│0│0│  ← Zero padding                          │
│   │0│ │ │ │ │ │0│                                          │
│   │0│ │Image │0│                                           │
│   │0│ │ │ │ │ │0│                                          │
│   │0│0│0│0│0│0│0│                                          │
│   └─┴─┴─┴─┴─┴─┴─┘                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Famous CNN Architectures

| Architecture | Year | Key Innovation |
|--------------|------|----------------|
| **LeNet** | 1998 | First successful CNN (handwritten digits) |
| **AlexNet** | 2012 | Deep CNN, ReLU, dropout (ImageNet breakthrough) |
| **VGG** | 2014 | Very deep with small 3×3 filters |
| **GoogLeNet/Inception** | 2014 | Inception modules (parallel filter sizes) |
| **ResNet** | 2015 | Skip connections enabling 100+ layers |
| **EfficientNet** | 2019 | Compound scaling of depth/width/resolution |

## Learn More
- **Visual**: [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
- **Course**: [CS231n Stanford (YouTube)](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)
- **Interactive**: [Convolution Visualizer](https://ezyang.github.io/convolution-visualizer/)
- **Paper**: [ResNet Paper](https://arxiv.org/abs/1512.03385)

---

# 7. Recurrent Neural Networks (RNNs)

## Overview
RNNs process sequential data by maintaining a hidden state that carries information across time steps. This makes them suitable for text, speech, time series, and any data where order matters.

## The Problem with Feedforward Networks

Feedforward networks treat each input independently — they have no memory. For sequences like "The cat sat on the ___", we need context from previous words.

## RNN Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RNN UNROLLED THROUGH TIME                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Same weights shared across all time steps                 │
│                                                             │
│        h₀          h₁          h₂          h₃              │
│   ──▶ ┌───┐  ──▶ ┌───┐  ──▶ ┌───┐  ──▶ ┌───┐  ──▶ h₄      │
│       │RNN│       │RNN│       │RNN│       │RNN│             │
│       └─┬─┘       └─┬─┘       └─┬─┘       └─┬─┘             │
│         │           │           │           │               │
│         ▲           ▲           ▲           ▲               │
│         │           │           │           │               │
│        x₁          x₂          x₃          x₄              │
│       "The"       "cat"       "sat"        "on"            │
│                                                             │
│   Hidden state hₜ = f(hₜ₋₁, xₜ)                            │
│   Carries information from past to future                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## The Vanishing Gradient Problem

In long sequences, gradients shrink exponentially when backpropagating through time. Information from early tokens gets "forgotten."

```
┌─────────────────────────────────────────────────────────────┐
│               VANISHING GRADIENT PROBLEM                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Gradient flow during backpropagation:                     │
│                                                             │
│   ←─────────────────────────────────────────────            │
│   Gradient gets smaller and smaller                         │
│                                                             │
│   t=100    t=50     t=10     t=5      t=1                  │
│   ┌───┐    ┌───┐    ┌───┐    ┌───┐    ┌───┐                │
│   │0.0│◀───│0.0│◀───│0.1│◀───│0.5│◀───│1.0│  ← Loss       │
│   └───┘    └───┘    └───┘    └───┘    └───┘                │
│     ↑        ↑        ↑                                     │
│   Can't    Can't   Barely                                   │
│   learn    learn   learns                                   │
│                                                             │
│   Early words have almost no influence on prediction!       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## LSTM (Long Short-Term Memory)

LSTMs solve vanishing gradients using gates that control information flow.

```
┌─────────────────────────────────────────────────────────────┐
│                      LSTM CELL                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    Cell State (memory highway)              │
│   ════════════════════●═══════════════●═════════════════▶  │
│                       │               │                     │
│                   ┌───┴───┐       ┌───┴───┐                 │
│                   │   ×   │       │   +   │                 │
│                   └───┬───┘       └───┬───┘                 │
│                       │               │                     │
│          ┌────────────┴───┐   ┌───────┴────────┐           │
│          │                │   │                │           │
│      ┌───┴───┐        ┌───┴───┴───┐        ┌───┴───┐       │
│      │Forget │        │  Input    │        │Output │       │
│      │ Gate  │        │  Gate     │        │ Gate  │       │
│      │  fₜ   │        │ iₜ × c̃ₜ  │        │  oₜ   │       │
│      └───┬───┘        └───┬───────┘        └───┬───┘       │
│          │                │                    │           │
│          └────────┬───────┴────────────────────┘           │
│                   │                                         │
│               ┌───┴───┐                                     │
│   hₜ₋₁ ─────▶│ LSTM  │─────▶ hₜ                            │
│               │ Cell  │                                     │
│   xₜ ───────▶└───────┘                                     │
│                                                             │
│   Gates (sigmoid 0-1):                                      │
│   • Forget gate: What to erase from cell state             │
│   • Input gate: What new info to add                       │
│   • Output gate: What to output to hidden state            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## GRU (Gated Recurrent Unit)

Simplified LSTM with 2 gates instead of 3. Often performs similarly with fewer parameters.

| Aspect | LSTM | GRU |
|--------|------|-----|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| Parameters | More | Fewer |
| Performance | Slightly better on complex tasks | Comparable, faster training |

## Bidirectional RNN

Processes sequence in both directions — useful when future context matters.

```
   Forward:  "The cat sat on the ___"  →  →  →
   Backward: "The cat sat on the ___"  ←  ←  ←
   
   Combined: Full context from both directions
```

## Learn More
- **Visual**: [Understanding LSTM (Colah's Blog)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- **Course**: [Stanford CS224n NLP](https://web.stanford.edu/class/cs224n/)
- **Interactive**: [RNN Effectiveness (Karpathy)](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- **Paper**: [Original LSTM Paper](https://www.bioinf.jku.at/publications/older/2604.pdf)

---

# 8. Transformers & Attention

## Overview
Transformers revolutionized NLP by replacing recurrence with attention mechanisms. They process entire sequences in parallel, capture long-range dependencies, and scale efficiently with compute.

## Why Attention?

RNNs process sequentially — slow and struggle with long sequences. Attention allows direct connections between any two positions.

```
┌─────────────────────────────────────────────────────────────┐
│                  RNN vs TRANSFORMER                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   RNN: Sequential processing                                │
│   ○──▶○──▶○──▶○──▶○──▶○                                    │
│   Word 1 must pass through all words to reach Word 6        │
│   (Information bottleneck, slow, hard to parallelize)       │
│                                                             │
│   TRANSFORMER: Parallel with attention                      │
│       ○───○                                                 │
│      /│\ /│\                                                │
│     / │ X │ \   Every word can attend to every other word  │
│    /  │/ \│  \  directly — no bottleneck!                  │
│   ○───○───○───○                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Self-Attention Mechanism

Each word creates three vectors: Query (Q), Key (K), Value (V)
- Query: "What am I looking for?"
- Key: "What do I contain?"
- Value: "What do I actually give?"

```
┌─────────────────────────────────────────────────────────────┐
│                    SELF-ATTENTION                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input: "The cat sat on the mat"                           │
│                                                             │
│   For the word "sat":                                       │
│                                                             │
│   Q_sat ──┐                                                 │
│           │     ┌─────────────────────────────────┐         │
│   K_The ──┼────▶│                                 │         │
│   K_cat ──┼────▶│  Attention = softmax(Q·Kᵀ/√d)  │         │
│   K_sat ──┼────▶│                                 │         │
│   K_on  ──┼────▶│  Weights:                       │         │
│   K_the ──┼────▶│  The: 0.05                      │         │
│   K_mat ──┘     │  cat: 0.60  ← "sat" attends    │         │
│                 │  sat: 0.15     most to "cat"   │         │
│                 │  on:  0.05                      │         │
│                 │  the: 0.05                      │         │
│                 │  mat: 0.10                      │         │
│                 └─────────────────────────────────┘         │
│                              │                              │
│                              ▼                              │
│   Output_sat = 0.05×V_The + 0.60×V_cat + 0.15×V_sat + ...  │
│                                                             │
│   "sat" now contains contextual information,                │
│   especially about what/who sat (the cat!)                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  MULTI-HEAD ATTENTION                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Instead of one attention, use multiple "heads"            │
│   Each head can focus on different relationships            │
│                                                             │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│   │  Head 1  │  │  Head 2  │  │  Head 3  │  │  Head 4  │   │
│   │ Subject- │  │  Object  │  │ Position │  │ Semantic │   │
│   │  Verb    │  │relations │  │  nearby  │  │ similar  │   │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│        │             │             │             │          │
│        └──────┬──────┴──────┬──────┘             │          │
│               │             │                    │          │
│               ▼             ▼                    ▼          │
│        ┌──────────────────────────────────────────┐        │
│        │              Concatenate                 │        │
│        │                  +                       │        │
│        │           Linear projection             │        │
│        └──────────────────────────────────────────┘        │
│                          │                                  │
│                          ▼                                  │
│                   Combined output                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Transformer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              TRANSFORMER ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│          ENCODER                        DECODER             │
│     (Understanding)                  (Generation)           │
│                                                             │
│   ┌─────────────────┐           ┌─────────────────┐        │
│   │                 │           │                 │        │
│   │  Feed Forward   │           │  Feed Forward   │        │
│   │                 │           │                 │        │
│   └────────┬────────┘           └────────┬────────┘        │
│            │                             │                  │
│   ┌────────┴────────┐           ┌────────┴────────┐        │
│   │   Add & Norm    │           │   Add & Norm    │        │
│   └────────┬────────┘           └────────┬────────┘        │
│            │                             │                  │
│   ┌────────┴────────┐           ┌────────┴────────┐        │
│   │                 │           │ Cross-Attention │        │
│   │ Self-Attention  │     ┌────▶│  (to encoder)   │        │
│   │                 │     │     │                 │        │
│   └────────┬────────┘     │     └────────┬────────┘        │
│            │              │              │                  │
│   ┌────────┴────────┐     │     ┌────────┴────────┐        │
│   │   Add & Norm    │─────┘     │   Add & Norm    │        │
│   └────────┬────────┘           └────────┬────────┘        │
│            │                             │                  │
│            │                    ┌────────┴────────┐        │
│            │                    │Masked Self-Attn │        │
│            │                    │ (causal)        │        │
│            │                    └────────┬────────┘        │
│            │                             │                  │
│   ┌────────┴────────┐           ┌────────┴────────┐        │
│   │ Input Embedding │           │Output Embedding │        │
│   │       +         │           │       +         │        │
│   │Positional Enc.  │           │Positional Enc.  │        │
│   └─────────────────┘           └─────────────────┘        │
│            ▲                             ▲                  │
│            │                             │                  │
│    "The cat sat"               "Le chat" (so far)          │
│                                                             │
│   ×N layers                     ×N layers                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Architecture Variants

| Type | Architecture | Examples | Use Case |
|------|--------------|----------|----------|
| **Encoder-Only** | Bidirectional attention | BERT, RoBERTa | Understanding, classification |
| **Decoder-Only** | Causal (masked) attention | GPT, Claude, Llama | Text generation |
| **Encoder-Decoder** | Full transformer | T5, BART, original | Translation, summarization |

## Positional Encoding

Transformers have no inherent sense of order. Positional encodings add position information.

```
Word embedding:     [0.2, 0.5, 0.1, ...]
Position encoding:  [0.0, 1.0, 0.0, ...]  (for position 0)
                  + ─────────────────────
Final:              [0.2, 1.5, 0.1, ...]
```

## Learn More
- **Essential**: [Attention Is All You Need Paper](https://arxiv.org/abs/1706.03762)
- **Visual**: [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
- **Interactive**: [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- **Deep Dive**: [Andrej Karpathy's GPT Video](https://www.youtube.com/watch?v=kCc8FmEb1nY)

---

# 9. Large Language Models (LLMs)

## Overview
LLMs are transformer-based models trained on massive text corpora. They learn statistical patterns of language and can perform a wide range of tasks through prompting, from writing code to answering questions.

## How LLMs Work

**Pre-training objective**: Predict the next token given all previous tokens

```
Input:  "The quick brown fox"
Target: "jumps"

P(jumps | The, quick, brown, fox) = ?
```

Through billions of such predictions, the model learns grammar, facts, reasoning patterns, and more.

## Key Concepts

| Term | Explanation |
|------|-------------|
| **Token** | Subword unit (e.g., "un", "believ", "able") — vocabulary typically 32K-100K tokens |
| **Context Window** | Maximum tokens the model can process at once (4K to 1M+) |
| **Embedding** | Dense vector representation of a token (dimension 1K-16K) |
| **Autoregressive** | Generates one token at a time, each conditioned on all previous |
| **Temperature** | Controls randomness: 0 = deterministic, 1+ = more random |
| **Top-p (nucleus)** | Only sample from tokens comprising top p% probability mass |
| **Top-k** | Only sample from k most likely tokens |

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM GENERATION                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Prompt: "The capital of France is"                        │
│                                                             │
│   Step 1: Process entire prompt                             │
│   ┌─────────────────────────────────────────┐              │
│   │  The → capital → of → France → is →     │              │
│   │                                    ↓    │              │
│   │                              [probability│              │
│   │                               distribution]             │
│   └─────────────────────────────────────────┘              │
│                                                             │
│   Step 2: Sample next token                                 │
│   ┌─────────────────────────────────────────┐              │
│   │ Paris: 0.85  ←── Most likely            │              │
│   │ Lyon:  0.03                              │              │
│   │ the:   0.02                              │              │
│   │ ...                                      │              │
│   └─────────────────────────────────────────┘              │
│                                                             │
│   Step 3: Append "Paris", repeat                            │
│   "The capital of France is Paris"                          │
│                                    ↓                        │
│   Step 4: Generate "."                                      │
│   "The capital of France is Paris."                         │
│                                    ↓                        │
│   Step 5: Generate <EOS> or continue...                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     TOKENIZATION                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Text: "I love transformers!"                              │
│                                                             │
│   Word-level:  [I] [love] [transformers] [!]  (4 tokens)   │
│                Problem: huge vocabulary, rare words         │
│                                                             │
│   Character:   [I][ ][l][o][v][e]...          (21 tokens)  │
│                Problem: too long, no semantic units         │
│                                                             │
│   Subword (BPE): [I] [love] [transform][ers][!] (5 tokens) │
│                Best of both: reasonable vocab, handles      │
│                rare words by breaking into pieces           │
│                                                             │
│   "unbelievable" → [un][believ][able]                      │
│   "transformers" → [transform][ers]                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                  LLM TRAINING PIPELINE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   STAGE 1: PRE-TRAINING                                     │
│   ┌─────────────────────────────────────────────────────┐  │
│   │  Massive text corpus (trillions of tokens)          │  │
│   │  • Web pages, books, code, Wikipedia                │  │
│   │  • Objective: Next-token prediction                 │  │
│   │  • Cost: $10M-$100M+ in compute                     │  │
│   └─────────────────────────────────────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│   ┌─────────────────────────────────────────────────────┐  │
│   │  BASE MODEL (can complete text, but not helpful)    │  │
│   └─────────────────────────────────────────────────────┘  │
│                          │                                  │
│   STAGE 2: SUPERVISED FINE-TUNING (SFT)                    │
│   ┌─────────────────────────────────────────────────────┐  │
│   │  High-quality instruction-response pairs            │  │
│   │  • "Explain quantum computing" → [good explanation] │  │
│   │  • Teaches model to be helpful assistant            │  │
│   └─────────────────────────────────────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│   ┌─────────────────────────────────────────────────────┐  │
│   │  SFT MODEL (helpful, but may have issues)           │  │
│   └─────────────────────────────────────────────────────┘  │
│                          │                                  │
│   STAGE 3: ALIGNMENT (RLHF/DPO)                            │
│   ┌─────────────────────────────────────────────────────┐  │
│   │  Human preference data                              │  │
│   │  • Compare response A vs B, human picks better      │  │
│   │  • Train reward model, optimize with RL             │  │
│   │  • Makes model helpful, harmless, honest            │  │
│   └─────────────────────────────────────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│   ┌─────────────────────────────────────────────────────┐  │
│   │  ALIGNED MODEL (Claude, ChatGPT, etc.)              │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Prompting Techniques

| Technique | Description | Example |
|-----------|-------------|---------|
| **Zero-shot** | No examples, just instruction | "Translate to French: Hello" |
| **Few-shot** | Include examples in prompt | "cat→chat, dog→chien, bird→?" |
| **Chain-of-Thought** | Ask for step-by-step reasoning | "Let's think step by step..." |
| **System prompt** | Set behavior/persona | "You are a helpful coding assistant" |

## Learn More
- **Visual**: [The Illustrated GPT-2 (Jay Alammar)](https://jalammar.github.io/illustrated-gpt2/)
- **Technical**: [GPT-3 Paper](https://arxiv.org/abs/2005.14165)
- **Course**: [Andrej Karpathy's LLM Course](https://karpathy.ai/zero-to-hero.html)
- **Guide**: [Anthropic's Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)

---

# 10. Generative AI

## Overview
Generative AI creates new content — text, images, audio, video, code — by learning patterns from training data. Unlike discriminative models that classify, generative models learn to produce.

## Generative vs Discriminative

```
┌─────────────────────────────────────────────────────────────┐
│            DISCRIMINATIVE vs GENERATIVE                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   DISCRIMINATIVE                   GENERATIVE               │
│   "What is this?"                  "Make me one of these"   │
│                                                             │
│   Input: 🖼️ (image)                Input: "a cat"           │
│          ↓                                ↓                 │
│   ┌─────────────┐                  ┌─────────────┐          │
│   │  Classifier │                  │  Generator  │          │
│   └──────┬──────┘                  └──────┬──────┘          │
│          ↓                                ↓                 │
│   Output: "cat" (95%)              Output: 🖼️ (new image)   │
│                                                             │
│   Learns: P(label | data)          Learns: P(data)          │
│   Examples: SVM, Random Forest     Examples: GAN, Diffusion │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Main Approaches

### 1. GANs (Generative Adversarial Networks)

Two networks competing: Generator creates fakes, Discriminator detects them.

```
┌─────────────────────────────