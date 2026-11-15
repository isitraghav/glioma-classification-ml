"""Build the 14-slide presentation (report/Final_Presentation.pptx) following the course slide structure.
Numbers come from the saved outputs of glioma_analysis.ipynb.  Run from project root: python report/build_slides.py
"""
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

FIG = "report/figs/"
OUT = "report/Final_Presentation.pptx"
NAVY, BLUE, INK, INK2, LIGHT, WHITE = (RGBColor(0x14, 0x2B, 0x4C), RGBColor(0x2A, 0x78, 0xD6), RGBColor(0x1A, 0x1A, 0x1A),
                                       RGBColor(0x52, 0x51, 0x4E), RGBColor(0xEE, 0xF3, 0xFA), RGBColor(0xFF, 0xFF, 0xFF))
ORANGE, AQUA = RGBColor(0xEB, 0x68, 0x34), RGBColor(0x1B, 0xAF, 0x7A)
FONT = "Calibri"
TITLE = "Brain Cancer Type Classification from Tumour Gene Activity Using Machine Learning"
TEAM = [("Raghav Yadav", "240598"), ("Darshita Yadav", "240981"), ("Prerit Shrivastava", "240593"),
        ("Saksham Lodha", "240626"), ("Rayna Manchanda", "240601")]

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
W, H = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]
slide_no = [0]


def text(slide, x, y, w, h, content, size=16, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         bullets=False, spacing=6):
    """content: str or list of str. '**' prefix -> bold line. Bullets get a • marker."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    lines = content if isinstance(content, list) else [content]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(spacing)
        b = bold
        sz = size
        if isinstance(line, tuple):
            line, sz = line
        if line.startswith("**"):
            line, b = line[2:], True
        if bullets and line:
            line = "•  " + line
        # inline **bold** segments
        parts = line.split("**")
        for k, part in enumerate(parts):
            if not part:
                continue
            r = p.add_run()
            r.text = part
            r.font.size, r.font.name, r.font.color.rgb = Pt(sz), FONT, color
            r.font.bold = b or (k % 2 == 1)
    return tb


def new_slide(title, subtitle=None, notes=None):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(1.0))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
    text(s, Inches(0.5), Inches(0.12), Inches(11.5), Inches(0.8), title, size=28, bold=True, color=WHITE,
         anchor=MSO_ANCHOR.MIDDLE)
    slide_no[0] += 1
    text(s, Inches(12.3), Inches(7.05), Inches(0.9), Inches(0.35), str(slide_no[0] - 1), size=11, color=INK2,
         align=PP_ALIGN.RIGHT)
    text(s, Inches(0.5), Inches(7.05), Inches(9), Inches(0.35), "Glioma type classification from gene expression  ·  CSE2032 Machine Learning",
         size=11, color=INK2)
    if subtitle:
        text(s, Inches(0.5), Inches(1.1), Inches(12.3), Inches(0.5), subtitle, size=16, color=INK2)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def table(slide, x, y, w, rows, col_w=None, size=12, row_h=0.36, header_fill=NAVY, bold_rows=()):
    nr, nc = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nr, nc, x, y, w, Inches(row_h * nr))
    t = shp.table
    if col_w:
        for j, cw in enumerate(col_w):
            t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame
            tf.word_wrap = True
            tf.paragraphs[0].text = ""
            r = tf.paragraphs[0].add_run()
            r.text = str(val)
            r.font.size, r.font.name = Pt(size), FONT
            if i == 0:
                r.font.bold, r.font.color.rgb = True, WHITE
                c.fill.solid(); c.fill.fore_color.rgb = header_fill
            else:
                r.font.color.rgb = INK
                r.font.bold = i in bold_rows
                c.fill.solid(); c.fill.fore_color.rgb = LIGHT if i % 2 == 0 else WHITE
    return t


def picture(slide, path, x, y, w=None, h=None):
    return slide.shapes.add_picture(path, x, y, width=w, height=h)


def box(slide, x, y, w, h, title, body, fill=LIGHT, title_color=NAVY, size=14):
    r = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    r.adjustments[0] = 0.06
    r.fill.solid(); r.fill.fore_color.rgb = fill; r.line.color.rgb = RGBColor(0xC9, 0xD6, 0xEA)
    text(slide, x + Inches(0.15), y + Inches(0.1), w - Inches(0.3), Inches(0.5), title, size=size + 3, bold=True, color=title_color)
    text(slide, x + Inches(0.15), y + Inches(0.6), w - Inches(0.3), h - Inches(0.7), body, size=size, bullets=True, spacing=5)


# ---------------------------------------------------------------- Slide 0: Title
s = prs.slides.add_slide(BLANK)
slide_no[0] += 1
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H); bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
acc = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(2.95), Inches(1.4), Inches(0.08)); acc.fill.solid(); acc.fill.fore_color.rgb = ORANGE; acc.line.fill.background()
text(s, Inches(0.6), Inches(0.7), Inches(12), Inches(0.5), "CSE2032 Machine Learning  ·  Project Evaluation  ·  November 2026", size=16, color=RGBColor(0xC9, 0xD6, 0xEA))
text(s, Inches(0.6), Inches(1.25), Inches(12), Inches(1.6), TITLE, size=36, bold=True, color=WHITE)
text(s, Inches(0.6), Inches(3.2), Inches(6.5), Inches(1.2),
     "Predicting the WHO 2021 molecular type of adult diffuse glioma (IDH / 1p19q) from tumour RNA-seq alone",
     size=18, color=RGBColor(0xDD, 0xE6, 0xF3))
t = table(s, Inches(7.6), Inches(3.2), Inches(5.1), [["Team member", "Enrollment No."]] + [list(m) for m in TEAM],
          col_w=[3.1, 2.0], size=15, row_h=0.45, header_fill=ORANGE)
text(s, Inches(0.6), Inches(6.5), Inches(12), Inches(0.5),
     "Department of Computer Science and Engineering  ·  School of Engineering and Technology  ·  BML Munjal University",
     size=14, color=RGBColor(0xC9, 0xD6, 0xEA))
s.notes_slide.notes_text_frame.text = ("Introduce the team. One line on the idea: doctors type gliomas with several molecular lab tests; "
                                       "we test whether one RNA-seq measurement can do the same job with machine learning.")

# ---------------------------------------------------------------- Slide 1: Objectives & deliverables
s = new_slide("Project Objectives and Deliverables", notes=(
    "Objectives: the classification task and the fair comparison. Deliverables: notebook and demo app. "
    "Mention that inputs are expression only; IDH and 1p/19q lab results define the label and are never inputs."))
box(s, Inches(0.5), Inches(1.4), Inches(6.0), Inches(4.6), "Objectives", [
    "Classify an adult diffuse glioma into its WHO 2021 molecular type (IDH-wildtype, IDH-mutant astrocytoma, "
    "IDH-mutant 1p/19q-codeleted oligodendroglioma) using only tumour gene expression.",
    "Compare a regularised baseline, simple models and bagging / boosting ensembles under one leakage-free, "
    "cross-validated protocol, and check that the model relies on real biology.",
], size=20)
box(s, Inches(6.85), Inches(1.4), Inches(6.0), Inches(4.6), "Deliverables", [
    "A fully reproducible Jupyter notebook: data merge, EDA, feature derivation, 5 tuned models, "
    "evaluation with 11 metrics and bootstrap confidence intervals.",
    "A working Streamlit demo app that predicts the glioma type of unseen patients (single patient, "
    "CSV upload, whole test set), plus this report and presentation.",
], size=20)
text(s, Inches(0.5), Inches(6.2), Inches(12.3), Inches(0.6),
     "Motivation: since WHO 2021, glioma type is defined by IDH mutation and 1p/19q codeletion [1]; each needs a separate lab test. "
     "One RNA-seq assay measures ~20,000 genes at once.", size=14, color=INK2)

# ---------------------------------------------------------------- Slide 2: Literature
s = new_slide("Literature Review: Earlier Work on the TCGA Glioma Data", notes=(
    "Early TCGA work was unsupervised (Verhaak, Brat, Ceccarelli). Supervised work usually predicts IDH or 1p/19q "
    "as two separate binary tasks, often from methylation or other assays. Our gap: the 3-class WHO decision from "
    "expression only, with five model families compared under one protocol. Results across papers are not directly comparable."))
table(s, Inches(0.4), Inches(1.25), Inches(12.5), [
    ["Study", "Data (same TCGA cohort unless noted)", "Task", "Method", "Key result"],
    ["Verhaak et al., 2010 [2]", "TCGA GBM expression, 202 tumours", "GBM subtypes", "Consensus clustering + ClaNC (840 genes)", "4 subtypes; proneural ↔ IDH1"],
    ["Brat et al. (TCGA), 2015 [3]", "TCGA LGG, 293 tumours, multi-omics", "LGG subtypes", "Integrative clustering", "IDH, 1p/19q, TP53 beat histology"],
    ["Ceccarelli et al., 2016 [4]", "TCGA LGG+GBM, 1,122 patients (our source)", "Pan-glioma subtypes", "Clustering + methylation classifier", "IDH/codel classes; Table S1 labels"],
    ["Wang et al., 2019 [5]", "TCGA RNA (692) + REMBRANDT (222)", "1p/19q codeletion", "GSVA, PAM, smoother, ...", "98.4% (TCGA), 98.6% (REMBRANDT)"],
    ["Chen et al., 2019 [6]", "TCGA RNA editing, 638 tumours", "IDH; 1p/19q", "SVM, RF, AdaBoost", "AUC 0.989 (IDH), 0.990 (1p/19q)"],
    ["Tasci et al., 2022 [7]", "TCGA + CGGA clinical & mutations", "LGG vs GBM grade", "Voting feature selection + ensemble", "87.6% (TCGA), 79.7% (CGGA)"],
    ["Munquad et al., 2022 [8]", "TCGA GBM RNA-seq + methylation", "GBM subtypes", "CNN", "CNN > classical ML"],
    ["This project", "TCGA LGG+GBM RNA-seq, 654 patients", "3-class IDH/1p19q type", "LR, DT, KNN, RF, XGBoost", "Test macro-F1 0.971 (RF), 0.984 (LR)"],
], col_w=[2.3, 3.0, 1.9, 2.7, 2.6], size=12, row_h=0.52, bold_rows=(8,))

# ---------------------------------------------------------------- Slide 3: Dataset
s = new_slide("Dataset Description", "TCGA lower-grade glioma + glioblastoma (Ceccarelli et al., Cell 2016), NCI Genomic Data Commons [4]",
              notes=("Two files, joined on the TCGA case ID. 974 labelled patients in Table S1, 667 with expression, 654 in both. "
                     "Features are 20,330 numeric gene expression values; target is the 3-class IDH/codel subtype. "
                     "Grade is only used in EDA."))
table(s, Inches(0.4), Inches(1.7), Inches(7.6), [
    ["Variable", "Data type", "Role / description"],
    ["Case (TCGA barcode)", "String (ID)", "Join key, e.g. TCGA-CS-4938"],
    ["20,330 gene columns", "Numeric, continuous ≥ 0", "Feature: RSEM-normalised expression [9]"],
    ["IDH/codel subtype", "Categorical, nominal (3)", "Target: IDHwt / IDHmut-non-codel / IDHmut-codel"],
    ["Grade", "Categorical, ordinal", "EDA only (G2, G3, G4), not a model input"],
], col_w=[2.2, 2.2, 3.2], size=13, row_h=0.55)
table(s, Inches(8.4), Inches(1.7), Inches(4.5), [
    ["Statistic", "Value"],
    ["Expression file", "667 patients × 20,330 genes"],
    ["Table S1", "1,122 patients (974 labelled)"],
    ["Final cohort (inner join)", "654 patients"],
    ["Missing values / duplicates", "0 / 0"],
], col_w=[2.3, 2.2], size=13, row_h=0.55)
table(s, Inches(0.4), Inches(4.8), Inches(7.6), [
    ["Target class", "WHO 2021 type [1]", "Patients"],
    ["IDHmut-non-codel", "Astrocytoma, IDH-mutant", "256 (39%)"],
    ["IDHwt", "Glioblastoma, IDH-wildtype", "229 (35%)"],
    ["IDHmut-codel", "Oligodendroglioma, IDH-mut + 1p/19q-codel", "169 (26%)"],
], col_w=[2.2, 3.8, 1.6], size=13, row_h=0.45)
picture(s, FIG + "cell13.png", Inches(8.9), Inches(4.4), h=Inches(2.55))

# ---------------------------------------------------------------- Slide 4: EDA & feature engineering
s = new_slide("Methodology (1/3): EDA and Feature Engineering", notes=(
    "Each EDA finding drove one design choice. Key point: every learned step (filter, scaler, PCA) is fitted on training "
    "data only, inside the pipeline, to avoid the selection bias described by Ambroise & McLachlan."))
table(s, Inches(0.4), Inches(1.25), Inches(7.9), [
    ["EDA finding", "Feature-engineering response"],
    ["p ≫ n: 20,330 genes vs 654 patients", "Strong regularisation; simple baselines"],
    ["Skew: max 884,605, median 519", "log2(x + 1) transform"],
    ["183 genes all-zero; 1,879 zero in > 90%", "Drop genes expressed in < 20% of training patients (→ 17,963)"],
    ["Classes 256 / 229 / 169", "Stratified 70/10/20 split; macro-F1 metric"],
    ["Scale-sensitive LR and KNN", "StandardScaler inside each pipeline"],
    ["Distances weak in high dimension", "PCA (10 / 30 components) for KNN"],
    ["PCA: IDHwt separate; IDHmut classes overlap", "Expect errors between IDH-mutant classes"],
    ["IDHwt mostly grade 4 (135/220)", "Grade excluded from inputs; noted as confounder"],
    ["Leakage risk [10]", "All learned steps fitted on training folds only"],
], col_w=[3.9, 4.0], size=13, row_h=0.5)
picture(s, FIG + "cell19.png", Inches(8.6), Inches(1.25), w=Inches(4.4))
picture(s, FIG + "cell17.png", Inches(8.5), Inches(4.65), w=Inches(4.6))

# ---------------------------------------------------------------- Slide 5: Baseline + comparison models
s = new_slide("Methodology (2/3): Baseline and Comparison Models", notes=(
    "Baseline is L2 logistic regression: the standard first model for high-dimensional expression data. "
    "The decision tree is the building block of both ensembles, so it measures what bagging and boosting add. "
    "KNN tests whether local similarity is enough."))
box(s, Inches(0.5), Inches(1.35), Inches(4.0), Inches(5.4), "Baseline: Logistic Reg.", [
    "Multinomial logistic regression with an L2 (ridge) penalty.",
    "Why: linear and hard to overfit once penalised, which suits p ≫ n [11].",
    "Coefficients are directly interpretable as gene weights.",
    "Usually well-calibrated probabilities.",
    "Tuned: C ∈ {0.0001 … 1} (inverse penalty strength).",
], size=18)
box(s, Inches(4.7), Inches(1.35), Inches(4.0), Inches(5.4), "Decision Tree (CART)", [
    "Why: transparent rules; the base learner of RF and XGBoost, so it is the reference for ensembles [12].",
    "Regularised by max depth, min leaf size and cost-complexity pruning.",
    "Tuned: depth {3, 5, 8, ∞}, leaf {1, 5, 15}, α {0, 0.005, 0.02}.",
], size=18)
box(s, Inches(8.9), Inches(1.35), Inches(4.0), Inches(5.4), "k-Nearest Neighbours", [
    "Why: non-parametric, instance-based; tests if similar tumours share a type [13].",
    "Runs on PCA components because raw distances fail in 18k dimensions.",
    "Tuned: PCA {10, 30}, k {3, 7, 15, 25}, uniform vs distance weights.",
], size=18)

# ---------------------------------------------------------------- Slide 6: Ensembles + tuning
s = new_slide("Methodology (3/3): Ensembles and Hyperparameter Tuning", notes=(
    "Random forest = bagging, reduces variance of trees. XGBoost = boosting, reduces bias. Grid search with 5-fold "
    "stratified CV on training patients only, macro-F1. The validation set picks the final model; the test set is used once."))
box(s, Inches(0.5), Inches(1.35), Inches(6.0), Inches(2.75), "Random Forest (bagging) [14]", [
    "Many decorrelated trees on bootstrap samples + random gene subsets; averaging cuts variance.",
    "Tuned: trees {200, 500}, max_features {√p, 0.2}, min leaf {1, 3, 8}.",
], size=17)
box(s, Inches(6.85), Inches(1.35), Inches(6.0), Inches(2.75), "XGBoost (boosting) [15]", [
    "Shallow trees added sequentially on the log-loss gradient; cuts bias.",
    "Tuned: trees {100, 300}, depth {2, 4}, learning rate {0.05, 0.2}, colsample {0.2, 0.5}, λ {1, 10}.",
], size=17)
table(s, Inches(0.5), Inches(4.35), Inches(12.35), [
    ["Step", "Protocol"],
    ["Tuning", "Exhaustive GridSearchCV, 5-fold stratified CV on 457 training patients, scored by macro-F1 [16]"],
    ["Model selection", "Highest macro-F1 on the 66-patient validation set → Random Forest"],
    ["Final evaluation", "131 untouched test patients, used once; 11 metrics + 1,000-sample bootstrap 95% CI"],
], col_w=[2.3, 10.05], size=14, row_h=0.55)

# ---------------------------------------------------------------- Slide 7: Results table
s = new_slide("Experimental Results: Held-out Test Set (131 patients)", notes=(
    "Selected RF: accuracy 0.969, macro-F1 0.971, CI 0.939-0.994. LR scored highest on test but was not the validation "
    "pick; we do not re-select after seeing the test set. CIs overlap: one patient = 0.8 points. "
    "Bagging clearly beats a single tree; boosting ties bagging but has the best probabilities (log-loss 0.077)."))
table(s, Inches(0.4), Inches(1.25), Inches(12.5), [
    ["Model", "Accuracy", "Bal. acc.", "Macro-F1 (95% CI)", "ROC-AUC", "PR-AUC", "MCC", "Log-loss", "Train acc.", "Gap"],
    ["Logistic Regression", "0.985", "0.987", "0.984 (0.957–1.000)", "0.992", "0.980", "0.977", "0.098", "1.000", "0.015"],
    ["Random Forest ★ (selected)", "0.969", "0.972", "0.971 (0.939–0.994)", "0.997", "0.993", "0.954", "0.135", "0.998", "0.028"],
    ["XGBoost", "0.969", "0.972", "0.970 (0.938–0.993)", "0.999", "0.998", "0.954", "0.077", "1.000", "0.031"],
    ["Decision Tree", "0.947", "0.952", "0.949 (0.908–0.984)", "0.956", "0.916", "0.919", "1.960", "0.958", "0.012"],
    ["KNN", "0.931", "0.935", "0.934 (0.892–0.971)", "0.972", "0.952", "0.896", "1.440", "1.000", "0.069"],
], col_w=[2.6, 1.0, 1.0, 2.1, 1.0, 1.0, 0.9, 1.0, 1.0, 0.9], size=13, row_h=0.48, bold_rows=(2,))
picture(s, FIG + "test_f1_ci.png", Inches(0.4), Inches(4.3), h=Inches(2.65))
box(s, Inches(6.3), Inches(4.3), Inches(6.6), Inches(2.65), "Takeaways", [
    "Bagging vs single tree: macro-F1 0.949 → 0.971; log-loss 1.96 → 0.14.",
    "Boosting ≈ bagging on labels; best probabilities (ROC-AUC 0.999).",
    "Penalised linear model is as good as any ensemble when p ≫ n.",
    "Top-3 CIs overlap: 1 patient ≈ 0.8 accuracy points.",
], size=14)

# ---------------------------------------------------------------- Slide 8: Regularisation + CV
s = new_slide("Regularisation and Cross-validated Results", notes=(
    "Every model chose strong regularisation: C=0.001, depth-3 trees with 15+ patients per leaf, depth-2 XGBoost "
    "with learning rate 0.05 and 20% of genes per tree. CV and test agree within ~1.5 points for the top models; "
    "RF's perfect validation score was partly luck. KNN overfits most (gap 0.069)."))
table(s, Inches(0.4), Inches(1.25), Inches(6.4), [
    ["Model", "Best (regularising) hyperparameters", "CV F1"],
    ["Log. Regression", "C = 0.001 (strong L2 penalty)", "0.983"],
    ["Decision Tree", "depth 3, min leaf 15, α = 0", "0.931"],
    ["KNN", "10 PCs, k = 3, distance weights", "0.953"],
    ["Random Forest", "200 trees, 20% genes/split, min leaf 8", "0.983"],
    ["XGBoost", "300 trees, depth 2, lr 0.05, colsample 0.2, λ = 1", "0.989"],
], col_w=[1.6, 3.9, 0.9], size=13, row_h=0.55)
table(s, Inches(0.4), Inches(4.8), Inches(6.4), [
    ["Model", "CV F1", "Validation F1", "Test F1"],
    ["Random Forest ★", "0.983", "1.000", "0.971"],
    ["Logistic Regression", "0.983", "0.986", "0.984"],
    ["XGBoost", "0.989", "0.986", "0.970"],
], col_w=[2.2, 1.4, 1.4, 1.4], size=13, row_h=0.45)
picture(s, FIG + "cv_val_test.png", Inches(7.0), Inches(1.3), w=Inches(6.0))
text(s, Inches(7.1), Inches(4.4), Inches(5.9), Inches(2.5), [
    "All five models picked their most conservative settings: with ~18,000 genes, many small weights generalise better than a few large ones.",
    "CV and test agree within ~1.5 points for the strong models, so the CV estimates were reliable.",
    "KNN has the largest train–test gap (0.069): it memorises training patients.",
], size=14, bullets=True)

# ---------------------------------------------------------------- Slide 9: Deployment diagram
s = new_slide("Deployment Architecture", "Deployed locally as a Streamlit web app; cloud deployment is planned future work",
              notes=("We did not deploy to a cloud platform (this slide is optional). The app runs locally: train_models.py "
                     "re-fits the tuned pipelines and saves them with joblib; app.py loads them and serves predictions in the browser."))
steps = [("TCGA data\n(GDC files)", "Expression matrix\n+ Table S1"), ("glioma_analysis.ipynb", "EDA, tuning,\nmodel selection"),
         ("train_models.py", "Re-fit 5 tuned\npipelines"), ("models.joblib", "Saved models +\n131 test patients"),
         ("app.py (Streamlit)", "log2 → 17,963 genes\n→ predict_proba"), ("Browser", "Prediction, WHO type,\nprobabilities")]
bw, gap, y0 = Inches(1.85), Inches(0.28), Inches(2.5)
for i, (t1, t2) in enumerate(steps):
    x = Inches(0.45) + i * (bw + gap)
    r = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y0, bw, Inches(2.0))
    r.fill.solid(); r.fill.fore_color.rgb = NAVY if i in (1, 4) else LIGHT
    r.line.color.rgb = BLUE
    col = WHITE if i in (1, 4) else NAVY
    text(s, x, y0 + Inches(0.2), bw, Inches(0.8), t1, size=13, bold=True, color=col, align=PP_ALIGN.CENTER)
    text(s, x, y0 + Inches(1.0), bw, Inches(0.9), t2, size=12, color=col if i in (1, 4) else INK2, align=PP_ALIGN.CENTER)
    if i < len(steps) - 1:
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x + bw + Inches(0.03), y0 + Inches(0.85), gap - Inches(0.06), Inches(0.3))
        a.fill.solid(); a.fill.fore_color.rgb = ORANGE; a.line.fill.background()
text(s, Inches(0.5), Inches(5.0), Inches(12.3), Inches(1.6), [
    "Run:  python app/train_models.py   →   python -m streamlit run app/app.py   →   http://localhost:8501",
    "The app's whole-test-set view reproduces the notebook's confusion matrix exactly (accuracy 0.969).",
    "Next step: host on Streamlit Community Cloud or Hugging Face Spaces (models.joblib is 3.2 MB).",
], size=15, bullets=True)

# ---------------------------------------------------------------- Slide 10: Demo screenshots
s = new_slide("Demonstration: Streamlit App", notes=(
    "Live demo order: 1) pick test patient TCGA-06-0184 -> IDHwt, 88.5%, correct. 2) Whole test set tab: confusion "
    "matrix, 4 errors, 3 of them at the IDHwt / IDHmut-non-codel boundary in grade 3-4 tumours. 3) Upload example CSV."))
picture(s, FIG + "demo_single.png", Inches(0.35), Inches(1.2), w=Inches(6.4))
picture(s, FIG + "demo_testset_crop.png", Inches(6.95), Inches(1.2), w=Inches(6.0))
text(s, Inches(0.35), Inches(5.45), Inches(6.4), Inches(1.5), [
    "Single held-out patient: predicted IDHwt (glioblastoma) with 88.5% confidence; lab label agrees.",
    "Model selector shows each model's test accuracy and macro-F1.",
], size=13, bullets=True)
text(s, Inches(6.95), Inches(5.45), Inches(6.0), Inches(1.5), [
    "Whole test set: 127/131 correct; IDHmut-codel never missed.",
    "3 of 4 errors are grade 3-4 tumours on the IDHwt / IDH-mutant boundary: the grade shortcut fails there.",
], size=13, bullets=True)

# ---------------------------------------------------------------- Slide 11: Team matrix (blank for the team to fill)
s = new_slide("Team Responsibility / Self-Contribution Matrix", notes="Each member explains their own contribution.")
acts = ["Data preprocessing & EDA", "Feature engineering", "Model building", "Model comparisons", "Ensemble",
        "Hyperparameter tuning / CV", "Demo app (Streamlit)", "Documentation (report)", "Presentation"]
table(s, Inches(0.4), Inches(1.3), Inches(12.5), [["Activity"] + [f"{n}\n({r})" for n, r in TEAM]] + [[a] + [""] * 5 for a in acts],
      col_w=[3.0, 1.9, 1.9, 1.9, 1.9, 1.9], size=13, row_h=0.52)

# ---------------------------------------------------------------- Slide 12: Conclusions
s = new_slide("Conclusions", notes=(
    "Answer the four research questions, then be honest about limits: single cohort, small evaluation sets, grade "
    "confounding. Highlight the biology: 8 of the top 10 genes sit on arms 1p/19q, consistent with (not proof of) codeletion."))
box(s, Inches(0.5), Inches(1.3), Inches(6.2), Inches(5.5), "Key outcomes", [
    "Expression alone recovers the 3 WHO glioma types: selected RF test accuracy 0.969, macro-F1 0.971 (CI 0.939–0.994).",
    "Strongly regularised logistic regression matched the ensembles (0.984): with p ≫ n, simple and penalised wins.",
    "Bagging beat a single tree; boosting tied bagging but gave the best probabilities.",
    "Biology: top gene RBP1 is silenced in IDH-mutant gliomas [17]; 8 of the top 10 genes lie on chromosome arms 1p/19q, a pattern consistent with the effects of 1p/19q codeletion.",
], size=18)
box(s, Inches(6.9), Inches(1.3), Inches(6.0), Inches(5.5), "Limitations and future work", [
    "One TCGA cohort, no external validation; next: CGGA [18] with batch correction [19].",
    "Small validation (66) and test (131) sets; next: repeated / nested CV and paired tests.",
    "Grade confounding: IDHwt mostly grade 4; next: evaluate within grade.",
    "Compact features (chromosome-arm and pathway scores); cloud deployment.",
    "Research demo only, not a validated diagnostic.",
], size=18, fill=RGBColor(0xFD, 0xF0, 0xEA), title_color=RGBColor(0x9A, 0x3A, 0x12))

# ---------------------------------------------------------------- Slide 13: References
s = new_slide("References", notes="Full reference list (38 entries) is in the project report.")
refs = [
    "[1] D. N. Louis et al., The 2021 WHO Classification of Tumors of the CNS: a summary, Neuro-Oncology 23(8):1231–1251, 2021.",
    "[2] R. G. W. Verhaak et al., Integrated genomic analysis identifies clinically relevant subtypes of glioblastoma, Cancer Cell 17(1):98–110, 2010.",
    "[3] Cancer Genome Atlas Research Network (D. J. Brat et al.), Comprehensive, integrative genomic analysis of diffuse lower-grade gliomas, NEJM 372:2481–2498, 2015.",
    "[4] M. Ceccarelli et al., Molecular profiling reveals biologically discrete subsets and pathways of progression in diffuse glioma, Cell 164(3):550–563, 2016.",
    "[5] Z.-L. Wang et al., Predicting chromosome 1p/19q codeletion by RNA expression profile, Aging 11(3):974–985, 2019.",
    "[6] S. C.-C. Chen et al., RNA editing-based classification of diffuse gliomas, BMC Bioinformatics 20(Suppl 19):659, 2019.",
    "[7] E. Tasci et al., Hierarchical voting-based feature selection and ensemble learning for glioma grading, Int. J. Mol. Sci. 23(22):14155, 2022.",
    "[8] S. Munquad et al., A deep learning-based framework for supporting clinical diagnosis of glioblastoma subtypes, Front. Genet. 13:855420, 2022.",
    "[9] B. Li and C. N. Dewey, RSEM: accurate transcript quantification from RNA-Seq data, BMC Bioinformatics 12:323, 2011.",
    "[10] C. Ambroise and G. J. McLachlan, Selection bias in gene extraction on the basis of microarray gene-expression data, PNAS 99(10):6562–6566, 2002.",
    "[11] S. le Cessie and J. C. van Houwelingen, Ridge estimators in logistic regression, Applied Statistics 41(1):191–201, 1992.",
    "[12] L. Breiman, J. Friedman, R. Olshen, C. Stone, Classification and Regression Trees, Wadsworth, 1984.",
    "[13] T. Cover and P. Hart, Nearest neighbor pattern classification, IEEE Trans. Inf. Theory 13(1):21–27, 1967.",
    "[14] L. Breiman, Random forests, Machine Learning 45(1):5–32, 2001.",
    "[15] T. Chen and C. Guestrin, XGBoost: a scalable tree boosting system, Proc. ACM SIGKDD, 785–794, 2016.",
    "[16] F. Pedregosa et al., Scikit-learn: machine learning in Python, JMLR 12:2825–2830, 2011.",
    "[17] A. P. Chou et al., Identification of RBP1 promoter hypermethylation in IDH1 and 2 mutant gliomas, J. Natl. Cancer Inst. 104(19):1458–1469, 2012.",
    "[18] Z. Zhao et al., Chinese Glioma Genome Atlas (CGGA), Genomics Proteomics Bioinformatics 19(1):1–12, 2021.",
    "[19] W. E. Johnson, C. Li, A. Rabinovic, Adjusting batch effects in microarray expression data using empirical Bayes methods, Biostatistics 8(1):118–127, 2007.",
]
text(s, Inches(0.5), Inches(1.2), Inches(12.3), Inches(5.8), [(r, 13) for r in refs], spacing=2)

prs.save(OUT)
print("written", OUT, "slides:", len(prs.slides))
