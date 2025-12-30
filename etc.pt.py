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

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Company Document Classifier",
    page_icon="📄",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
MODEL_PATH = r"C:\DESKTOP\5th SEM EL\company document classify\bert_company_model"
LABEL_ENCODER_PATH = r"C:\DESKTOP\5th SEM EL\company document classify\bert_company_model\label_encoder.pkl"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
label_encoder = pickle.load(open(LABEL_ENCODER_PATH, "rb"))

# ---------------- UTIL FUNCTIONS ----------------
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return " ".join(page.extract_text() or "" for page in reader.pages)

def predict_with_confidence(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)
    label = label_encoder.inverse_transform([pred.item()])[0]
    return label, confidence.item() * 100

def render_pdf_page(file, page_no, zoom=1.5):
    file.seek(0)
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    page = pdf[page_no]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def document_bot(question, text, predicted_class):
    q = question.lower()
    if "type" in q or "document" in q:
        return f"This document is classified as **{predicted_class}**."
    if "date" in q or "due" in q or "payment" in q:
        dates = re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)
        return f"I found these dates in the document: {', '.join(dates[:5])}" if dates else "No payment date found."
    if "invoice" in q or "order" in q:
        ids = re.findall(r"\b\d{5,}\b", text)
        return f"Possible document numbers: {', '.join(ids[:5])}" if ids else "No invoice/order number found."
    if "amount" in q or "total" in q:
        amounts = re.findall(r"\₹?\$?\s?\d+[.,]?\d*", text)
        return f"Possible amounts found: {', '.join(amounts[:5])}" if amounts else "No amount detected."
    return "I can help with document type, dates, invoice numbers, and amounts."

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
/* General */
body {
    background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
    font-family: 'Segoe UI', sans-serif;
}
/* Header */
h1 {
    color: #2c3e50;
    font-size: 3em;
}
p {
    font-size: 1.2em;
}
/* Metrics */
div[data-testid="metric-container"] {
    background: #ffffff70 !important;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
}
/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #ff7e5f, #feb47b);
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px 20px;
}
/* Slider */
.css-1aumxhk {
    color: #ff7e5f;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HERO SECTION ----------------
st.markdown(
    """
    <div style='text-align:center; padding:20px; background:linear-gradient(90deg, #ff7e5f, #feb47b); border-radius:20px; color:white;'>
        <h1>📄 Company Document Classifier</h1>
        <p>AI-powered document understanding with smart assistant</p>
    </div>
    """,
    unsafe_allow_html=True
)

add_vertical_space(2)

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📤 Upload your PDF document",
    type="pdf",
    help="Only PDF files are supported"
)

if uploaded_file:
    with st.spinner("🔍 Analyzing document..."):
        text = extract_text_from_pdf(uploaded_file)
        label, confidence = predict_with_confidence(text)

    # ---------------- RESULTS ----------------
    st.markdown("### 📊 Document Analysis Results")
    col1, col2 = st.columns(2)
    col1.metric("📌 Document Type", label)
    col2.metric("📈 Confidence", f"{confidence:.2f}%")

    add_vertical_space(2)

    # ---------------- PDF PREVIEW ----------------
    st.markdown("### 📄 PDF Preview")
    reader = PyPDF2.PdfReader(uploaded_file)
    total_pages = len(reader.pages)
    page = st.slider("Select page", 1, total_pages, 1)
    img = render_pdf_page(uploaded_file, page - 1)
    st.image(img, use_container_width=True)

    add_vertical_space(2)

    # ---------------- DOCUMENT CHATBOT ----------------
    st.markdown("### 🤖 Ask Document Assistant")
    user_q = st.text_input(
        "Ask something about the document:",
        placeholder="What is the document type? What is the due date?"
    )

    if user_q:
        answer = document_bot(user_q, text, label)
        st.success(answer)

# ---------------- FOOTER ----------------
st.markdown(
    "<p style='text-align:center;color:gray; padding:20px;'>Built with ❤️ using BERT & Streamlit</p>",
    unsafe_allow_html=True
)
