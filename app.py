import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import requests
import urllib.parse
import os
import gdown
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

def generate_pdf_report(smiles, compound_name, mol, pred_label, proba):
    img_path = "/tmp/mol_structure_report.png"
    Draw.MolToImage(mol, size=(350, 350)).save(img_path)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "P2X7 Activity Prediction Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 8, f"Compound Name: {compound_name or 'Not found in PubChem'}")

    # Manually break the SMILES into fixed-width lines using cell() instead of multi_cell(),
    # since multi_cell's auto word-wrap fails on strings with no spaces to break on.
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
    pdf.cell(0, 8, f"Probability of Active: {proba:.1%}", new_x="LMARGIN", new_y="NEXT")
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

    col_widths = [70, 35, 30, 35]
    headers = ["SMILES", "Compound Name", "Predicted", "Prob. Active"]

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(220, 220, 220)
    for h, w in zip(headers, col_widths):
        pdf.cell(w, 8, h, border=1, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", size=8)
    for _, row in results_df.iterrows():
        smi = str(row.get("smiles", ""))[:35]
        name = str(row.get("compound_name", "") or "-")[:20]
        pred = str(row.get("predicted_class", ""))
        proba = row.get("probability_active", None)
        proba_str = f"{proba:.1%}" if pd.notna(proba) else "-"

        if pdf.get_y() > 270:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(220, 220, 220)
            for h, w in zip(headers, col_widths):
                pdf.cell(w, 8, h, border=1, fill=True)
            pdf.ln()
            pdf.set_font("Helvetica", size=8)

        pdf.cell(col_widths[0], 7, smi, border=1)
        pdf.cell(col_widths[1], 7, name, border=1)
        pdf.cell(col_widths[2], 7, pred, border=1)
        pdf.cell(col_widths[3], 7, proba_str, border=1)
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

with st.sidebar:
    st.header("About P2X7")
    st.markdown(
        "**P2X7** is an ATP-gated ion channel receptor found on immune cells (like macrophages "
        "and microglia) and several other tissue types. When activated by high extracellular ATP "
        "(a signal often released by damaged or dying cells), it opens a pore that allows ion flux "
        "into the cell."
    )
    st.markdown(
        "**Toxicity link:** Sustained or excessive P2X7 activation is associated with "
        "inflammatory and cell-death pathways — implicated in chronic inflammation, "
        "neurodegeneration, and certain inflammatory diseases."
    )
    st.markdown(
        "**What this model does:** predicts whether a compound is likely to be active at the "
        "P2X7 receptor (IC50 ≤ 1000 nM), based on its molecular structure."
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

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])

with tab1:
    smiles_input = st.text_input("Enter a SMILES string:", placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O")
    st.markdown(
        "🔗 Don't have a SMILES string? Look up your compound on "
        "[PubChem](https://pubchem.ncbi.nlm.nih.gov/) and copy its **Canonical SMILES** from the compound page."
    )
    if smiles_input.strip():
        X, mol = featurize(smiles_input.strip())
        if X is None:
            st.error("Invalid SMILES string — could not parse this molecule.")
        else:
            proba = model.predict_proba(X)[0, 1]
            pred = int(proba >= 0.5)
            pred_label = "Active (IC50 ≤ 1000 nM)" if pred == 1 else "Inactive (IC50 > 1000 nM)"
            with st.spinner("Looking up compound name on PubChem..."):
                compound_name = get_compound_name(smiles_input.strip())
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Structure")
                st.image(Draw.MolToImage(mol, size=(300, 300)))
                st.markdown(f"**Compound name:** {compound_name if compound_name else '_Not found in PubChem_'}")
            with col2:
                st.subheader("Prediction")
                if pred == 1:
                    st.success(f"**{pred_label}**")
                else:
                    st.info(f"**{pred_label}**")
                st.metric("Probability of Active", f"{proba:.1%}")
                st.progress(float(proba))
                pdf_bytes = generate_pdf_report(smiles_input.strip(), compound_name, mol, pred_label, proba)
                st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name="p2x7_prediction_report.pdf", mime="application/pdf")
    else:
        st.info("Enter a SMILES string above to get started.")

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
                        row = {"smiles": smi, "compound_name": None, "predicted_class": "Invalid SMILES", "probability_active": None}
                    else:
                        proba = model.predict_proba(X)[0, 1]
                        pred = "Active" if proba >= 0.5 else "Inactive"
                        name = get_compound_name(str(smi)) if fetch_names else None
                        row = {"smiles": smi, "compound_name": name, "predicted_class": pred, "probability_active": round(float(proba), 4)}
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
