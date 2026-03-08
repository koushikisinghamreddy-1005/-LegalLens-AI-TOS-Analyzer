import streamlit as st
import torch
import torch.nn as nn
import joblib
import json
import PyPDF2
import io
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from deep_translator import GoogleTranslator

# --- 1. MODEL ARCHITECTURE (Matches your saved .pt file) ---
class LegalBERTWithClassifier(nn.Module):
    def __init__(self, base_model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :] 

# --- 2. CACHED RESOURCE LOADING ---
@st.cache_resource
def load_resources():
    # Load labels index
    with open('labels.json', 'r') as f:
        unique_labels = json.load(f)
    
    # Load BERT & SVM components
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    model = LegalBERTWithClassifier("nlpaueb/legal-bert-base-uncased", len(unique_labels))
    model.load_state_dict(torch.load('legalbert_model.pt', map_location=torch.device('cpu')))
    model.eval()
    svm_pipeline = joblib.load('svm_pipeline.pkl')
    
    # Load Summarizer Model manually (To avoid Task errors in Python 3.14)
    s_model_name = "sshleifer/distilbart-cnn-12-6"
    s_tokenizer = AutoTokenizer.from_pretrained(s_model_name)
    s_model = AutoModelForSeq2SeqLM.from_pretrained(s_model_name)
    
    return tokenizer, model, svm_pipeline, unique_labels, s_tokenizer, s_model

# Category Map (Modify based on your training data if needed)
LABEL_MAP = {
    0: "Arbitration: Right to go to court is waived.",
    1: "Unilateral Change: Company can change terms anytime.",
    2: "Content Removal: They can delete your account/data.",
    3: "Jurisdiction: Legal disputes happen in their city/country.",
    4: "Choice of Law: Contract governed by laws they choose.",
    5: "Limitation of Liability: They aren't responsible for damages.",
    6: "Unilateral Termination: They can ban you without cause.",
    7: "Contract by Use: You agree just by using the website."
}

# --- 3. UI STYLING ---
st.set_page_config(page_title="LegalLens Pro", page_icon="⚖️", layout="wide")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .stApp { background-color: #fcfcfc; }
    .risk-card { 
        background-color: #fff5f5; padding: 15px; border-radius: 8px; 
        border-left: 5px solid #ff4b4b; margin-bottom: 10px;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    .safe-card { 
        background-color: #f0fff4; padding: 15px; border-radius: 8px; 
        border-left: 5px solid #48bb78; margin-bottom: 10px;
    }
    .summary-box { 
        background-color: #ebf8ff; padding: 20px; border-radius: 10px; 
        border: 1px solid #90cdf4; font-size: 1.1rem;
    }
    .main-title { color: #2c5282; font-weight: 800; font-size: 2.5rem; text-align: center; margin-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True) # FIXED: changed unsafe_allow_index to unsafe_allow_html

# --- 4. NAVIGATION & INPUTS ---
st.markdown("<h1 class='main-title'>⚖️ LegalLens AI: TOS Analyzer</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3252/3252906.png", width=80)
    st.title("Settings")
    target_lang = st.selectbox("Select Language", ["English", "Hindi", "Spanish", "French", "Arabic", "German"])
    lang_codes = {"English":"en", "Hindi":"hi", "Spanish":"es", "French":"fr", "Arabic":"ar", "German":"de"}
    st.divider()
    st.caption("AI-Powered Transparency for Terms of Service Agreements.")

tab1, tab2 = st.tabs(["📝 Paste Text", "📂 Upload PDF/Document"])

final_text = ""

with tab1:
    paste_text = st.text_area("Paste Agreement Text:", height=250, placeholder="Paste TOS content here...")
    if paste_text: final_text = paste_text

with tab2:
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted: final_text += extracted
        else:
            final_text = str(uploaded_file.read(), "utf-8")
        st.success(f"File loaded: {uploaded_file.name}")

# --- 5. EXECUTION ---
if st.button("Analyze Legal Risks"):
    if not final_text.strip():
        st.error("Please provide an agreement to analyze.")
    else:
        tokenizer, model, svm_pipeline, unique_labels, s_tokenizer, s_model = load_resources()
        l_code = lang_codes[target_lang]

        with st.spinner('AI is performing a deep legal audit...'):
            # Translate Input to English (For AI)
            eng_text = GoogleTranslator(source='auto', target='en').translate(final_text[:4000])

            # 1. Risk Prediction (BERT + SVM)
            inputs = tokenizer(eng_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                embedding = model(inputs['input_ids'], inputs['attention_mask']).numpy()
            
            probs = svm_pipeline.predict_proba(embedding)[0]
            detected = [LABEL_MAP.get(unique_labels[i], f"Risk {unique_labels[i]}") for i, p in enumerate(probs) if p > 0.5]

            # 2. Summarization (Direct Generation)
            inputs_s = s_tokenizer([eng_text[:1024]], max_length=1024, return_tensors="pt")
            summary_ids = s_model.generate(inputs_s["input_ids"], num_beams=4, max_length=100)
            summary_en = s_tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]

            # 3. Final Translation (For User)
            trans_summary = GoogleTranslator(source='en', target=l_code).translate(summary_en)

            # --- DISPLAY RESULTS ---
            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"📖 Executive Summary ({target_lang})")
                st.markdown(f"<div class='summary-box'>{trans_summary}</div>", unsafe_allow_html=True)
                
                # Dynamic Safety Score
                score = max(100 - (len(detected) * 12), 0)
                st.write("")
                st.metric(label="Calculated Safety Score", value=f"{score}/100")

            with col2:
                st.subheader(f"🚩 Risk Detection ({target_lang})")
                if not detected:
                    st.markdown("<div class='safe-card'>✅ Safe: No major unfair clauses detected.</div>", unsafe_allow_html=True)
                else:
                    for risk in detected:
                        trans_risk = GoogleTranslator(source='en', target=l_code).translate(risk)
                        st.markdown(f"<div class='risk-card'><b>⚠️ {trans_risk}</b></div>", unsafe_allow_html=True)

st.divider()
st.caption("LegalLens AI Tool | For informational purposes only. Consult a lawyer for legal matters.")