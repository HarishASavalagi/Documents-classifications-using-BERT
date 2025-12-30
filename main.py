import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import PyPDF2
import fitz
import pickle
import re
from PIL import Image
from streamlit_extras.add_vertical_space import add_vertical_space
import pytesseract
from bs4 import BeautifulSoup
import io
import pandas as pd

# ---------------- OCR PATH (WINDOWS) ----------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Company Document Classifier",
    page_icon="📄",
    layout="wide"
)

# ---------------- PATHS ----------------
MODEL_PATH = r"C:\DESKTOP\5th SEM EL\company document classify\bert_company_model"
LABEL_ENCODER_PATH = r"C:\DESKTOP\5th SEM EL\company document classify\bert_company_model\label_encoder.pkl"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
label_encoder = pickle.load(open(LABEL_ENCODER_PATH, "rb"))

# ---------------- TEXT EXTRACTION ----------------
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return " ".join(page.extract_text() or "" for page in reader.pages)

def extract_text_from_image(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image)

def extract_text_from_html(file):
    html = file.read().decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator=" ")

def extract_text(file):
    name = file.name.lower()
    file.seek(0)

    if name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif name.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        return extract_text_from_image(file)
    elif name.endswith(".html"):
        return extract_text_from_html(file)
    else:
        return ""

# ---------------- TOP-K PREDICTION ----------------
def predict_top_k(text, k=3):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, k)

    labels = label_encoder.inverse_transform(top_idxs.cpu().numpy())
    confidences = (top_probs.cpu().numpy() * 100)

    return labels, confidences

# ---------------- PREVIEW PDF ----------------
def render_pdf_page(file, page_no, zoom=1.4):
    file.seek(0)
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    page = pdf[page_no]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

# ---------------- DOCUMENT BOT ----------------
def document_bot(question, text, predicted_class):
    q = question.lower()

    if "type" in q:
        return f"This document is classified as **{predicted_class}**."
    if "date" in q:
        dates = re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)
        return f"Dates found: {', '.join(dates[:5])}" if dates else "No dates found."
    if "invoice" in q:
        ids = re.findall(r"\b\d{5,}\b", text)
        return f"Invoice numbers: {', '.join(ids[:5])}" if ids else "No invoice numbers found."
    if "amount" in q:
        amounts = re.findall(r"\₹?\$?\s?\d+[.,]?\d*", text)
        return f"Amounts found: {', '.join(amounts[:5])}" if amounts else "No amounts found."

    return "I can help with document type, dates, invoice numbers, and amounts."

# ---------------- STYLING ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #eef2f3, #d9e4f5);
    font-family: 'Segoe UI', sans-serif;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}
.card-title {
    font-size: 1.3em;
    font-weight: bold;
}
.gradient-header {
    background: linear-gradient(90deg, #667eea, #764ba2);
    padding: 25px;
    border-radius: 20px;
    color: white;
    text-align: center;
}
div[data-testid="metric-container"] {
    background: #f8f9ff;
    border-radius: 14px;
    padding: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="gradient-header">
    <h1>📄 Company Document Classifier</h1>
    <p>PDF • Image • HTML • OCR • Confidence Visualization</p>
</div>
""", unsafe_allow_html=True)

add_vertical_space(2)

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "📤 Upload documents",
    type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "html"],
    accept_multiple_files=True
)

results = []

if uploaded_files:
    st.markdown("## 📊 Analysis Dashboard")

    for file in uploaded_files:
        with st.spinner(f"Analyzing {file.name}..."):
            text = extract_text(file)
            labels, confidences = predict_top_k(text, k=3)

        main_label = labels[0]
        main_conf = confidences[0]

        results.append({
            "file": file,
            "text": text,
            "label": main_label
        })

        st.markdown(f"""
        <div class="card">
            <div class="card-title">📄 {file.name}</div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        c1.metric("Document Type", main_label)
        c2.metric("Confidence", f"{main_conf:.2f}%")

        # 🔥 CONFIDENCE BAR CHART
        with st.expander("📊 Prediction Confidence (Top-3)"):
            df = pd.DataFrame({
                "Class": labels,
                "Confidence (%)": confidences
            })
            st.bar_chart(df, x="Class", y="Confidence (%)")

        # Preview
        if file.name.lower().endswith(".pdf"):
            with st.expander("📄 Preview PDF"):
                st.image(render_pdf_page(file, 0), use_container_width=True)
        elif file.name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            with st.expander("🖼 Preview Image"):
                st.image(file, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- CHATBOT ----------------
if results:
    st.markdown("## 🤖 Document Assistant")

    doc_names = [r["file"].name for r in results]
    selected_doc = st.selectbox("Select document", doc_names)

    selected = next(r for r in results if r["file"].name == selected_doc)

    user_q = st.text_input(
        "Ask a question:",
        placeholder="What is the document type? What is the invoice amount?"
    )

    if user_q:
        response = document_bot(user_q, selected["text"], selected["label"])
        st.success(response)

# ---------------- FOOTER ----------------
st.markdown(
    "<p style='text-align:center;color:gray;margin-top:40px;'>Built with ❤️ using BERT & Streamlit</p>",
    unsafe_allow_html=True
)
