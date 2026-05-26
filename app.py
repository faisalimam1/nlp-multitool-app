import streamlit as st
import torch
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
from keybert import KeyBERT

st.set_page_config(page_title="NLP Multi-Tool", page_icon="🤗", layout="centered")

@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis",
                    model="faisalimam19/bert-imdb-sentiment", device=-1)

@st.cache_resource
def load_bart():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

@st.cache_resource
def load_keywords():
    return KeyBERT(model="all-MiniLM-L6-v2")

@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification",
                    model="facebook/bart-large-mnli", device=-1)

def summarize(text, max_length=80, min_length=30):
    tokenizer, model = load_bart()
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    with torch.no_grad():
        ids = model.generate(inputs["input_ids"],
                             max_length=max_length, min_length=min_length,
                             do_sample=False, early_stopping=True)
    return tokenizer.decode(ids[0], skip_special_tokens=True)

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.title("🤗 NLP Multi-Tool")
    st.markdown("**Phase 2 Capstone — Day 16**")
    st.markdown("---")
    task = st.radio("Choose a task:", [
        "Sentiment Analysis",
        "Summarization",
        "Keyword Extraction",
        "Zero-Shot Classification"
    ])
    st.markdown("---")
    if task == "Sentiment Analysis":
        st.info("Model: faisalimam19/bert-imdb-sentiment\nFine-tuned BERT · F1: 0.921")
    elif task == "Summarization":
        st.info("Model: facebook/bart-large-cnn\nBART encoder-decoder")
    elif task == "Keyword Extraction":
        st.info("Model: all-MiniLM-L6-v2\nKeyBERT · Cosine similarity")
    else:
        st.info("Model: facebook/bart-large-mnli\nZero-shot via NLI")
    st.markdown("---")
    st.markdown("**Built by Faisal**")
    st.markdown("[GitHub](https://github.com/faisalimam1) · [HuggingFace](https://huggingface.co/faisalimam19)")

# ── Main ───────────────────────────────────────────────────
icon = {"Sentiment Analysis":"📊","Summarization":"📝",
        "Keyword Extraction":"🔑","Zero-Shot Classification":"🏷️"}
st.title(f"{icon[task]} {task}")

text_input = st.text_area("Paste your text here:", height=180,
                           placeholder="Enter any text to analyze...")

# ── Sentiment ──────────────────────────────────────────────
if task == "Sentiment Analysis":
    if st.button("Analyze", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing..."):
                result = load_sentiment()(text_input[:512])[0]
            label, score = result["label"], result["score"]
            if label == "POSITIVE":
                st.success(f"**{label}** — {score:.2%} confidence")
            else:
                st.error(f"**{label}** — {score:.2%} confidence")
            st.progress(score)
            st.markdown("---")
            st.markdown("#### 📊 Benchmark")
            st.table({
                "Model": ["faisalimam19/bert-imdb-sentiment (mine)", "distilbert-sst-2 (baseline)"],
                "Trained On": ["IMDB 20k reviews", "SST-2 snippets"],
                "F1": ["0.921", "~0.910"],
            })
            with st.expander("⚠️ Known Limitation — Sarcasm"):
                st.write("BERT reads word semantics, not intent. "
                         "'Truly masterful... put me to sleep' → POSITIVE 0.976. "
                         "Token-level models cannot detect sarcasm reliably.")

# ── Summarization ──────────────────────────────────────────
elif task == "Summarization":
    col1, col2 = st.columns(2)
    max_len = col1.slider("Max words", 30, 150, 80)
    min_len = col2.slider("Min words", 10, 60, 30)
    if st.button("Summarize", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter some text.")
        elif len(text_input.split()) < 30:
            st.warning("Need at least 30 words to summarize.")
        else:
            with st.spinner("Summarizing..."):
                summary = summarize(text_input[:1024], max_len, min_len)
            st.success(summary)
            orig = len(text_input.split())
            summ = len(summary.split())
            c1, c2, c3 = st.columns(3)
            c1.metric("Original", f"{orig} words")
            c2.metric("Summary",  f"{summ} words")
            c3.metric("Compression", f"{(1-summ/orig)*100:.0f}%")

# ── Keywords ───────────────────────────────────────────────
elif task == "Keyword Extraction":
    col1, col2 = st.columns(2)
    top_n = col1.slider("Keywords", 3, 10, 6)
    ngram = col2.selectbox("Phrase length",
                           ["Single words", "1-2 word phrases", "1-3 word phrases"])
    ngram_map = {"Single words":(1,1), "1-2 word phrases":(1,2), "1-3 word phrases":(1,3)}
    if st.button("Extract Keywords", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Extracting..."):
                kws = load_keywords().extract_keywords(
                    text_input, keyphrase_ngram_range=ngram_map[ngram],
                    stop_words="english", top_n=top_n)
            for word, score in kws:
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"**{word}**")
                c2.progress(score, text=f"{score:.3f}")
            st.caption("Uses same cosine similarity as Day 14 semantic search.")

# ── Zero-Shot ──────────────────────────────────────────────
elif task == "Zero-Shot Classification":
    labels_input = st.text_input("Labels (comma separated):",
                                  value="technology, politics, sports, healthcare, finance")
    if st.button("Classify", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            labels = [l.strip() for l in labels_input.split(",") if l.strip()]
            if len(labels) < 2:
                st.warning("Enter at least 2 labels.")
            else:
                with st.spinner("Classifying..."):
                    result = load_classifier()(text_input[:512], candidate_labels=labels)
                for label, score in zip(result["labels"], result["scores"]):
                    st.markdown(f"**{label}**")
                    st.progress(score, text=f"{score:.2%}")
                st.caption("No training needed. Define any labels — model uses NLI to classify.")
