"""Glioma subtype classifier - live demo.

Run from the project root:
    python app/train_models.py      # once, creates app/artifacts/
    streamlit run app/app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ART = Path(__file__).parent / "artifacts"
WHO = {
    "IDHwt": "Glioblastoma, IDH-wildtype",
    "IDHmut-non-codel": "Astrocytoma, IDH-mutant (1p/19q intact)",
    "IDHmut-codel": "Oligodendroglioma, IDH-mutant and 1p/19q-codeleted",
}

st.set_page_config(page_title="Glioma Subtype Classifier", page_icon="🧠", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(ART / "models.joblib")


@st.cache_data
def load_test():
    return pd.read_csv(ART / "test_patients.csv.gz", index_col=0)


bundle = load_bundle()
models, genes, classes = bundle["models"], bundle["genes"], bundle["classes"]
test = load_test()

st.title("🧠 Glioma subtype classification from tumour gene expression")
st.caption("Predicts the IDH/1p19q molecular subtype of an adult diffuse glioma from ~18,000 RNA-seq gene "
           "expression values (TCGA LGG+GBM, Ceccarelli et al., 2016). Research demo, not a diagnostic tool.")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", list(models), index=0,
                              help="Logistic Regression was selected on the validation set.")
    s = bundle["scores"][model_name]
    st.metric("Held-out test accuracy", f"{s['test_accuracy']:.3f}")
    st.metric("Held-out test macro-F1", f"{s['test_macro_f1']:.3f}")
    st.caption(f"{len(genes):,} genes used · 131 unseen test patients")

tab_one, tab_upload = st.tabs(["Single test patient", "Upload expression file"])


def show_prediction(row_log: pd.DataFrame, true_label=None):
    m = models[model_name]
    proba = m.predict_proba(row_log[genes])[0]
    pred = classes[int(np.argmax(proba))]
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.subheader("Prediction")
        st.markdown(f"### {pred}")
        st.write(f"**WHO 2021 equivalent:** {WHO[pred]}")
        st.write(f"**Confidence:** {proba.max():.1%}")
        if true_label is not None:
            if true_label == pred:
                st.success(f"Correct - lab-confirmed label is **{true_label}**")
            else:
                st.error(f"Incorrect - lab-confirmed label is **{true_label}**")
    with c2:
        st.subheader("Class probabilities")
        st.bar_chart(pd.DataFrame({"probability": proba}, index=classes), horizontal=True, height=220)


with tab_one:
    st.write("Pick one of the 131 held-out test patients. The model never saw these patients during training or tuning.")
    pid = st.selectbox("TCGA patient", test.index, index=0)
    row = test.loc[[pid]]
    st.write(f"WHO grade: **{row['grade'].iloc[0]}**")
    show_prediction(row, true_label=row["true_label"].iloc[0])
    with st.expander("Marker genes for this patient (log2 expression)"):
        markers = [g for g in ["CHI3L1", "PDPN", "EMP3", "RBP1", "TMEM167B", "EGFR", "OLIG2"] if g in genes]
        st.dataframe(row[markers].T.rename(columns={pid: "log2(RSEM+1)"}).round(2))

with tab_upload:
    st.write("Upload a CSV with one row per tumour and one column per gene (raw RSEM-normalised values, "
             "same gene names as the TCGA file). The app applies the same log2(x+1) transform. "
             "Genes missing from the file are set to 0.")
    st.caption(f"Example file: {ART / 'example_upload_raw.csv'}")
    up = st.file_uploader("Expression CSV", type=["csv"])
    if up is not None:
        raw = pd.read_csv(up, index_col=0)
        missing = [g for g in genes if g not in raw.columns]
        X = np.log2(raw.reindex(columns=genes, fill_value=0).clip(lower=0) + 1)
        if missing:
            st.warning(f"{len(missing):,} of {len(genes):,} model genes are missing and were set to 0.")
        proba = models[model_name].predict_proba(X)
        out = pd.DataFrame(proba, index=raw.index, columns=classes).round(3)
        out.insert(0, "prediction", [classes[i] for i in proba.argmax(1)])
        st.dataframe(out, use_container_width=True)
        st.download_button("Download predictions", out.to_csv().encode(), "predictions.csv", "text/csv")
