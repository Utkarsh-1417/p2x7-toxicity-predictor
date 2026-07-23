import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import requests
import urllib.parse
import os
import gdown
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys, Draw
from rdkit.ML.Descriptors import MoleculeDescriptors
from fpdf import FPDF

st.set_page_config(page_title="P2X7 Activity Predictor", page_icon="🧪", layout="wide")

# --- Download model files from Google Drive on first run ---
MODEL_FILE_ID = "1x4auGj_tLKnJgrUuznviPo7SUzAFgDmZ"
SCALER_FILE_ID = "1qizBoEogo6zg7u-rWlrZDkXp6wtb2Qtz"
CONFIG_FILE_ID = "1ZN8q2qgQIfn6SGuNSSVIFgvxkGn1OvFj"

os.makedirs("model_files", exist_ok=True)

@st.cache_resource
def load_artifacts():
    if not os.path.exists("model_files/p2x7_voting_ensemble_v2.pkl"):
        gdown.download(id=MODEL_FILE_ID, output="model_files/p2x7_voting_ensemble_v2.pkl", quiet=False)
    if not os.path.exists("model_files/descriptor_scaler.pkl"):
        gdown.download(id=SCALER_FILE_ID, output="model_files/descriptor_scaler.pkl", quiet=False)
    if not os.path.exists("model_files/feature_config.json"):
        gdown.download(id=CONFIG_FILE_ID, output="model_files/feature_config.json", quiet=False)

    model = joblib.load("model_files/p2x7_voting_ensemble_v2.pkl")
    scaler = joblib.load("model_files/descriptor_scaler.pkl")
    with open("model_files/feature_config.json") as f:
        config = json.load(f)
    return model, scaler, config

with st.spinner("Loading model (first load may take a minute)..."):
    model, scaler, config = load_artifacts()

def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp_morgan = np.array(morgan_gen.GetFingerprint(mol)).reshape(1, -1)
    fp_maccs = np.array(MACCSkeys.GenMACCSKeys(mol)).reshape(1, -1)
    descriptor_names = config['descriptors']
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    desc = np.array(calc.CalcDescriptors(mol)).reshape(1, -1)
    desc_scaled = scaler.transform(desc)
    X = np.hstack([fp_morgan, fp_maccs, desc_scaled])
    return X, mol

def get_compound_name(smiles, timeout=8):
    try:
        smiles_enc = urllib.parse.quote(smiles)
        cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles_enc}/cids/TXT"
        r = requests.get(cid_url, timeout=timeout)
        if r.status_code != 200 or not r.text.strip():
            return None
        cid = r.text.strip().split('\n')[0]
        syn_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/TXT"
        r2 = requests.get(syn_url, timeout=timeout)
        if r2.status_code != 200 or not r2.text.strip():
            return None
        return r2.text.strip().split('\n')[0]
    except requests.exceptions.RequestException:
        return None

def confidence_label(proba):
    """Confidence based on distance from the 0.5 decision boundary."""
    distance = abs(proba - 0.5)
    if distance >= 0.35:
        return "High confidence", "🟢"
    elif distance >= 0.15:
        return "Moderate confidence", "🟡"
    else:
        return "Low confidence (borderline)", "🔴"

def generate_pdf_report(smiles, compound_name, mol, pred_label, proba, conf_text):
    img_path = "/tmp/mol_structure_report.png"
    Draw.MolToImage(mol, size=(350, 350)).save(img_path)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "P2X7 Activity Prediction Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 8, f"Compound Name: {compound_name or 'Not found in PubChem'}")

    pdf.set_font("Helvetica", size=9)
    chunk_size = 45
    smiles_chunks = [smiles[i:i+chunk_size] for i in range(0, len(smiles), chunk_size)] or [""]
    pdf.cell(0, 6, f"SMILES: {smiles_chunks[0]}", new_x="LMARGIN", new_y="NEXT")
    for chunk in smiles_chunks[1:]:
        pdf.cell(0, 6, "  " + chunk, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.ln(3)

    pdf.image(img_path, x=70, w=70)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, f"Predicted Class: {pred_label}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 8, f"Probability Active: {proba:.1%}   |   Probability Inactive: {1-proba:.1%}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Confidence: {conf_text}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 6,
        "Disclaimer: This model is intended for experimental and educational use only. "
        "It is a screening tool and its predictions must not be used as a sole basis for "
        "any real-world chemical, pharmacological, or safety decision."
    )
    pdf.ln(8)

    pdf.set_font("Helvetica", size=8)
    pdf.cell(0, 5, "Built as a learning project at IIT (BHU) Varanasi under the AI in Drug Discovery Internship Program 2026", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 5, "Model Developed By: Ritul Kumari  |  Web App Developed By: Utkarsh Kumar", new_x="LMARGIN", new_y="NEXT", align="C")

    return bytes(pdf.output())

def generate_batch_pdf_report(results_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "P2X7 Batch Prediction Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(2)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, f"Total compounds: {len(results_df)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    col_widths = [65, 30, 25, 28, 32]
    headers = ["SMILES", "Compound", "Predicted", "Prob. Active", "Confidence"]

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(220, 220, 220)
    for h, w in zip(headers, col_widths):
        pdf.cell(w, 8, h, border=1, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", size=7)
    for _, row in results_df.iterrows():
        smi = str(row.get("smiles", ""))[:32]
        name = str(row.get("compound_name", "") or "-")[:16]
        pred = str(row.get("predicted_class", ""))
        proba = row.get("probability_active", None)
        proba_str = f"{proba:.1%}" if pd.notna(proba) else "-"
        conf = str(row.get("confidence", "-") or "-")[:18]

        if pdf.get_y() > 270:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_fill_color(220, 220, 220)
            for h, w in zip(headers, col_widths):
                pdf.cell(w, 8, h, border=1, fill=True)
            pdf.ln()
            pdf.set_font("Helvetica", size=7)

        pdf.cell(col_widths[0], 7, smi, border=1)
        pdf.cell(col_widths[1], 7, name, border=1)
        pdf.cell(col_widths[2], 7, pred, border=1)
        pdf.cell(col_widths[3], 7, proba_str, border=1)
        pdf.cell(col_widths[4], 7, conf, border=1)
        pdf.ln()

    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 5,
        "Disclaimer: This model is intended for experimental and educational use only. "
        "It is a screening tool and its predictions must not be used as a sole basis for "
        "any real-world chemical, pharmacological, or safety decision."
    )
    pdf.ln(4)
    pdf.set_font("Helvetica", size=7)
    pdf.cell(0, 5, "Built as a learning project at IIT (BHU) Varanasi under the AI in Drug Discovery Internship Program 2026", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 5, "Model Developed By: Ritul Kumari  |  Web App Developed By: Utkarsh Kumar", new_x="LMARGIN", new_y="NEXT", align="C")

    return bytes(pdf.output())

# --- Session state for history ---
if "history" not in st.session_state:
    st.session_state["history"] = []

def log_to_history(smiles, compound_name, pred_label, proba, conf_text):
    st.session_state["history"].append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "smiles": smiles,
        "compound_name": compound_name or "-",
        "predicted_class": pred_label,
        "probability_active": round(float(proba), 4),
        "probability_inactive": round(float(1 - proba), 4),
        "confidence": conf_text,
    })

# --- Sidebar ---
with st.sidebar:
    st.header("About P2X7")

    with st.expander("1. What is the P2X7 receptor? Where is it located?", expanded=True):
        st.markdown(
            "P2X7 is an **ATP-gated ion channel receptor** — it opens in response to high levels "
            "of extracellular ATP, rather than the usual chemical neurotransmitters. It is expressed "
            "mainly on **immune cells** (macrophages, microglia, dendritic cells, lymphocytes), and is "
            "also found in the **central nervous system, skin, bone, and epithelial tissues**."
        )

    with st.expander("2. What is its role in the body?"):
        st.markdown(
            "Under normal conditions, P2X7 acts as a **danger sensor**. Healthy cells keep very "
            "little ATP outside them, so a sudden rise in extracellular ATP (released by "
            "stressed, damaged, or dying cells) signals injury or infection. P2X7 detects this "
            "and triggers immune responses — including activation of the **inflammasome**, "
            "release of inflammatory cytokines (like IL-1β), and immune cell signaling."
        )

    with st.expander("3. How does P2X7 toxicity affect the body?"):
        st.markdown(
            "Problems arise when P2X7 is **activated too strongly or for too long**. This can "
            "trigger excessive inflammation and a cell-death pathway, contributing to "
            "**chronic inflammatory conditions, neurodegeneration, neuropathic pain, and tissue "
            "damage**. This dual nature — essential defense signal vs. driver of harmful "
            "inflammation when overactive — is why compounds affecting P2X7 need careful "
            "evaluation."
        )

    with st.expander("4. Basic principle of this model"):
        st.markdown(
            "The model learns from **thousands of known compounds** (BindingDB) whose real, "
            "measured P2X7 activity (IC50) is known. Each molecule is converted into numeric "
            "**structural fingerprints and physicochemical properties**, and an ensemble of "
            "machine learning models learns the patterns separating **active** (IC50 ≤ 1000 nM) "
            "from **inactive** compounds — outputting a probability, not a lab result."
        )

    st.divider()
    st.caption(f"Model: {config['final_model']}")

st.title("🧪 P2X7 Activity Predictor")

st.warning(
    "⚠️ **Disclaimer:** This model is intended for **experimental and educational use only**. "
    "It is a screening tool and its predictions must not be used as a sole basis for any "
    "real-world chemical, pharmacological, or safety decision. Always confirm findings "
    "experimentally with qualified professionals."
)

tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction (CSV)", f"History ({len(st.session_state['history'])})"])

# ============== TAB 1: SINGLE PREDICTION ==============
with tab1:
    if "smiles_box_input" not in st.session_state:
        st.session_state["smiles_box_input"] = ""
    if "single_result" not in st.session_state:
        st.session_state["single_result"] = None

    def set_example(smi):
        st.session_state["smiles_box_input"] = smi
        st.session_state["single_result"] = None

    st.markdown("**Try an example:**")
    examples = {
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
        "Propranolol": "CC(C)NCC(O)COc1cccc2ccccc12",
    }
    example_cols = st.columns(len(examples))
    for col, (name, smi) in zip(example_cols, examples.items()):
        with col:
            st.button(name, use_container_width=True, on_click=set_example, args=(smi,))

    smiles_input = st.text_input(
        "Enter a SMILES string:",
        placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O",
        key="smiles_box_input"
    )
    st.markdown(
        "🔗 Don't have a SMILES string? Look up your compound on "
        "[PubChem](https://pubchem.ncbi.nlm.nih.gov/) and copy its **Canonical SMILES** from the compound page."
    )

    predict_clicked = st.button("🔍 Predict", type="primary")

    if predict_clicked:
        if not smiles_input.strip():
            st.warning("Please enter a SMILES string first.")
            st.session_state["single_result"] = None
        else:
            X, mol = featurize(smiles_input.strip())
            if X is None:
                st.error("Invalid SMILES string — could not parse this molecule.")
                st.session_state["single_result"] = None
            else:
                proba = model.predict_proba(X)[0, 1]
                pred = int(proba >= 0.5)
                pred_label = "Active (IC50 ≤ 1000 nM)" if pred == 1 else "Inactive (IC50 > 1000 nM)"
                conf_text, conf_emoji = confidence_label(proba)
                with st.spinner("Looking up compound name on PubChem..."):
                    compound_name = get_compound_name(smiles_input.strip())
                st.session_state["single_result"] = {
                    "smiles": smiles_input.strip(),
                    "mol": mol,
                    "compound_name": compound_name,
                    "pred_label": pred_label,
                    "proba": proba,
                    "conf_text": conf_text,
                    "conf_emoji": conf_emoji,
                }
                log_to_history(smiles_input.strip(), compound_name, pred_label, proba, conf_text)

    result = st.session_state["single_result"]
    if result is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Structure")
            st.image(Draw.MolToImage(result["mol"], size=(300, 300)))
            st.markdown(f"**Compound name:** {result['compound_name'] if result['compound_name'] else '_Not found in PubChem_'}")
        with col2:
            st.subheader("Prediction")
            if "Active (" in result["pred_label"]:
                st.success(f"**{result['pred_label']}**")
            else:
                st.info(f"**{result['pred_label']}**")

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Probability Active", f"{result['proba']:.1%}")
            with m2:
                st.metric("Probability Inactive", f"{1 - result['proba']:.1%}")

            st.progress(float(result["proba"]))
            st.markdown(f"**Confidence:** {result['conf_emoji']} {result['conf_text']}")

            pdf_bytes = generate_pdf_report(result["smiles"], result["compound_name"], result["mol"], result["pred_label"], result["proba"], result["conf_text"])
            single_csv = pd.DataFrame([{
                "smiles": result["smiles"],
                "compound_name": result["compound_name"],
                "predicted_class": result["pred_label"],
                "probability_active": round(float(result["proba"]), 4),
                "probability_inactive": round(float(1 - result["proba"]), 4),
                "confidence": result["conf_text"],
            }]).to_csv(index=False).encode('utf-8')

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name="p2x7_prediction_report.pdf", mime="application/pdf", use_container_width=True)
            with dl_col2:
                st.download_button("📊 Download CSV", data=single_csv, file_name="p2x7_prediction.csv", mime="text/csv", use_container_width=True)
    elif not predict_clicked:
        st.info("Enter a SMILES string (or pick an example) and click Predict.")

# ============== TAB 2: BATCH PREDICTION ==============
with tab2:
    st.markdown("Upload a CSV with a column named **`smiles`** to predict activity for many molecules at once.")
    fetch_names = st.checkbox("Also fetch compound names from PubChem (slower)", value=False)
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        if 'smiles' not in df_input.columns:
            st.error("CSV must contain a column named 'smiles'.")
        else:
            with st.spinner(f"Predicting activity for {len(df_input)} molecules..."):
                results = []
                for smi in df_input['smiles']:
                    X, mol = featurize(str(smi))
                    if X is None:
                        row = {"smiles": smi, "compound_name": None, "predicted_class": "Invalid SMILES", "probability_active": None, "confidence": None}
                    else:
                        proba = model.predict_proba(X)[0, 1]
                        pred = "Active" if proba >= 0.5 else "Inactive"
                        conf_text, _ = confidence_label(proba)
                        name = get_compound_name(str(smi)) if fetch_names else None
                        row = {"smiles": smi, "compound_name": name, "predicted_class": pred, "probability_active": round(float(proba), 4), "confidence": conf_text}
                        log_to_history(str(smi), name, pred, proba, conf_text)
                    results.append(row)
                results_df = pd.DataFrame(results)

            st.success(f"Done — {len(results_df)} molecules processed.")
            st.dataframe(results_df, use_container_width=True)

            csv_out = results_df.to_csv(index=False).encode('utf-8')
            batch_pdf_bytes = generate_batch_pdf_report(results_df)

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button("📥 Download results as CSV", data=csv_out, file_name="p2x7_predictions.csv", mime="text/csv", use_container_width=True)
            with dl_col2:
                st.download_button("📄 Download results as PDF", data=batch_pdf_bytes, file_name="p2x7_batch_report.pdf", mime="application/pdf", use_container_width=True)

# ============== TAB 3: PREDICTION HISTORY ==============
with tab3:
    st.markdown("All predictions made during this session (single + batch). History clears when the app restarts or tab closes.")

    if len(st.session_state["history"]) == 0:
        st.info("No predictions yet this session. Try one in the Single Prediction or Batch Prediction tab.")
    else:
        history_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(history_df, use_container_width=True)

        hist_col1, hist_col2 = st.columns([1, 1])
        with hist_col1:
            hist_csv = history_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download full history as CSV", data=hist_csv, file_name="p2x7_session_history.csv", mime="text/csv", use_container_width=True)
        with hist_col2:
            if st.button("🗑️ Clear history", use_container_width=True):
                st.session_state["history"] = []
                st.rerun()

st.divider()
st.markdown(
    "<div style='text-align: center;'>"
    "<span style='font-size: 1.15em; font-weight: 600; color: #1a1a1a; background-color: #FFD54F; padding: 6px 14px; border-radius: 6px; display: inline-block;'>"
    "Built as a learning project at IIT (BHU) Varanasi under the AI in Drug Discovery Internship Program 2026"
    "</span>"
    "<br><br>"
    "<span style='font-size: 1.1em; font-weight: 600; color: #1a1a1a; background-color: #81E6B4; padding: 5px 12px; border-radius: 6px; display: inline-block;'>"
    "Model Developed By: Ritul Kumari"
    "</span>"
    "<br><br>"
    "<span style='font-size: 1.1em; font-weight: 600; color: #1a1a1a; background-color: #81E6B4; padding: 5px 12px; border-radius: 6px; display: inline-block;'>"
    "Web App Developed By: Utkarsh Kumar"
    "</span>"
    "</div>",
    unsafe_allow_html=True
)
