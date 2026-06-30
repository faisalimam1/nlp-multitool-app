# 🤗 NLP Multi-Tool App
### 4 NLP tasks. 1 app. Your own fine-tuned model in production.

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-nlp--multitool--app.streamlit.app-FF4B4B?style=for-the-badge)](https://nlp-multitool-app.streamlit.app)
[![HuggingFace](https://img.shields.io/badge/🤗%20Model-faisalimam19%2Fbert--imdb--sentiment-FFD21E?style=for-the-badge)](https://huggingface.co/faisalimam19/bert-imdb-sentiment)
[![GitHub](https://img.shields.io/badge/GitHub-faisalimam1-181717?style=for-the-badge&logo=github)](https://github.com/faisalimam1)

---

## 🎯 What This App Does

Paste any text. Choose a task. Get instant NLP results — powered by state-of-the-art HuggingFace models, including a BERT model I fine-tuned myself.

| Task | Model | Result |
|------|-------|--------|
| 📊 **Sentiment Analysis** | `faisalimam19/bert-imdb-sentiment` | Positive/Negative + confidence score |
| 📝 **Summarization** | `facebook/bart-large-cnn` | Compressed summary + compression % |
| 🔑 **Keyword Extraction** | `all-MiniLM-L6-v2` + KeyBERT | Top keywords ranked by semantic relevance |
| 🏷️ **Zero-Shot Classification** | `facebook/bart-large-mnli` | Classify into any labels — no training needed |

---

## 🧠 The Fine-Tuned Model

The sentiment task uses a BERT model I trained from scratch on Day 12 of my AI Engineer roadmap.

| | My Model | Baseline |
|--|---------|----------|
| **Model** | `faisalimam19/bert-imdb-sentiment` | `distilbert-sst-2` |
| **Trained On** | IMDB (20,000 reviews) | SST-2 snippets |
| **F1 Score** | **0.921** | ~0.910 |
| **Training Time** | ~28 min on T4 GPU | Pretrained |
| **Epochs** | 3 | — |

> Fine-tuning on domain-specific data (IMDB movie reviews) gives stronger results on that domain than a generic pretrained model.

---

## 🔬 How Each Task Works Internally

### 📊 Sentiment Analysis
BERT reads the full text bidirectionally — every token attends to every other token via self-attention. The `[CLS]` token's final embedding is passed through a linear classifier to output Positive/Negative probabilities.

**Known limitation:** Sarcasm breaks it. *"Truly masterful filmmaking. It put me to sleep."* → POSITIVE (0.976). BERT reads word semantics, not intent.

### 📝 Summarization
BART uses an encoder-decoder architecture. The encoder reads the entire document. The decoder generates the summary token by token. Trained on CNN/DailyMail news articles — works best on factual, structured text.

### 🔑 Keyword Extraction
KeyBERT embeds the entire document into a vector, then embeds each candidate word/phrase. Keywords = phrases whose embeddings have highest cosine similarity to the document embedding. Same cosine similarity used in semantic search — finding meaning, not just frequency.

### 🏷️ Zero-Shot Classification
Uses Natural Language Inference (NLI). For each label, it asks: *"Does this text entail it belongs to [label]?"* No training data required. Define any labels at inference time — the model has never seen them before.

---

## 📊 Live Test Results

| Task | Input | Output |
|------|-------|--------|
| Sentiment | *"The Dark Knight is a masterpiece..."* | POSITIVE — 99.60% |
| Summarization | 125-word AI article | 38-word summary — 70% compression |
| Keywords | Deep learning paragraph | neural, recognition, attention, convolutional... |
| Zero-Shot | Climate change article | climate change — 96.40% |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)

```
Streamlit          → Web app framework
HuggingFace        → Transformers, model hub, pipelines
PyTorch            → Model inference
KeyBERT            → Keyword extraction via embeddings
Sentence-Transformers → all-MiniLM-L6-v2 embeddings
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/faisalimam1/nlp-multitool-app.git
cd nlp-multitool-app
pip install -r requirements.txt
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## 📁 Project Structure

```
nlp-multitool-app/
├── app.py              ← Streamlit app (all 4 NLP tasks)
├── requirements.txt    ← Dependencies
└── README.md
```

---

## 🗺️ Part of the 30-Day AI Engineer Roadmap

This is the **Phase 2 Capstone** (Day 16) of my public AI Engineer roadmap.

| Phase | Topic | Status |
|-------|-------|--------|
| Phase 1 | Deep Learning Foundations | ✅ Complete |
| **Phase 2** | **HuggingFace & Transformers** | **✅ Complete** |
| Phase 3 | LLMs & RAG | ✅ Complete |
| Phase 4 | Deploy & Ship | ✅ Complete |

📌 Full roadmap: [github.com/faisalimam1/DL-Learning-Roadmap](https://github.com/faisalimam1/DL-Learning-Roadmap)

---

## 🔗 Connect

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Follow%20My%20Journey-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/faisalimam19)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebooks-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/faisalimam19)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/faisalimam19)

---

*Every number in this README was actually measured — not estimated.*
