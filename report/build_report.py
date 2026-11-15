"""Fill the course template (report/Final_Project_Report_Template.docx) with the project report.

Output: report/Final_Project_Report.docx (and .pdf via build step in __main__).
Every number is taken from the saved outputs of glioma_analysis.ipynb.
Run from the project root:  python report/build_report.py
"""
import copy
import re
import sys

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.text.paragraph import Paragraph

TEMPLATE = "report/Final_Project_Report_Template.docx"
OUT = "report/Final_Project_Report.docx"
FIG = "report/figs/"
FONT = "Cambria"
TITLE = "Brain Cancer Type Classification from Tumour Gene Activity Using Machine Learning"
TEAM = [("Raghav Yadav", "240598"), ("Darshita Yadav", "240981"), ("Prerit Shrivastava", "240593"),
        ("Saksham Lodha", "240626"), ("Rayna Manchanda", "240601")]

# ----------------------------------------------------------------------------- references (IEEE style)
REFS = [
    ("louis", 'D. N. Louis et al., "The 2021 WHO Classification of Tumors of the Central Nervous System: a summary," Neuro-Oncology, vol. 23, no. 8, pp. 1231-1251, 2021, doi: 10.1093/neuonc/noab106.'),
    ("ceccarelli", 'M. Ceccarelli et al., "Molecular profiling reveals biologically discrete subsets and pathways of progression in diffuse glioma," Cell, vol. 164, no. 3, pp. 550-563, 2016, doi: 10.1016/j.cell.2015.12.028.'),
    ("brat", 'Cancer Genome Atlas Research Network (D. J. Brat et al.), "Comprehensive, integrative genomic analysis of diffuse lower-grade gliomas," N. Engl. J. Med., vol. 372, no. 26, pp. 2481-2498, 2015, doi: 10.1056/NEJMoa1402121.'),
    ("verhaak", 'R. G. W. Verhaak et al., "Integrated genomic analysis identifies clinically relevant subtypes of glioblastoma characterized by abnormalities in PDGFRA, IDH1, EGFR, and NF1," Cancer Cell, vol. 17, no. 1, pp. 98-110, 2010, doi: 10.1016/j.ccr.2009.12.020.'),
    ("yan", 'H. Yan et al., "IDH1 and IDH2 mutations in gliomas," N. Engl. J. Med., vol. 360, no. 8, pp. 765-773, 2009, doi: 10.1056/NEJMoa0808710.'),
    ("noushmehr", 'H. Noushmehr et al., "Identification of a CpG island methylator phenotype that defines a distinct subgroup of glioma," Cancer Cell, vol. 17, no. 5, pp. 510-522, 2010, doi: 10.1016/j.ccr.2010.03.017.'),
    ("capper", 'D. Capper et al., "DNA methylation-based classification of central nervous system tumours," Nature, vol. 555, pp. 469-474, 2018, doi: 10.1038/nature26000.'),
    ("wang", 'Z.-L. Wang, Z. Zhao, Z. Wang, C.-B. Zhang and T. Jiang, "Predicting chromosome 1p/19q codeletion by RNA expression profile: a comparison of current prediction models," Aging, vol. 11, no. 3, pp. 974-985, 2019, doi: 10.18632/aging.101795.'),
    ("chen19", 'S. C.-C. Chen, C.-M. Lo, S.-H. Wang and E. C.-Y. Su, "RNA editing-based classification of diffuse gliomas: predicting isocitrate dehydrogenase mutation and chromosome 1p/19q codeletion," BMC Bioinformatics, vol. 20, suppl. 19, art. 659, 2019, doi: 10.1186/s12859-019-3236-0.'),
    ("tasci", 'E. Tasci, Y. Zhuge, H. Kaur, K. Camphausen and A. V. Krauze, "Hierarchical voting-based feature selection and ensemble learning model scheme for glioma grading with clinical and molecular characteristics," Int. J. Mol. Sci., vol. 23, no. 22, art. 14155, 2022, doi: 10.3390/ijms232214155.'),
    ("munquad", 'S. Munquad, T. Si, S. Mallik, A. B. Das and Z. Zhao, "A deep learning-based framework for supporting clinical diagnosis of glioblastoma subtypes," Front. Genet., vol. 13, art. 855420, 2022, doi: 10.3389/fgene.2022.855420.'),
    ("chou", 'A. P. Chou et al., "Identification of retinol binding protein 1 promoter hypermethylation in isocitrate dehydrogenase 1 and 2 mutant gliomas," J. Natl. Cancer Inst., vol. 104, no. 19, pp. 1458-1469, 2012, doi: 10.1093/jnci/djs357.'),
    ("gdc", 'NCI Genomic Data Commons, "LGG-GBM 2016 publication page (Ceccarelli et al., Cell 2016)," https://gdc.cancer.gov/about-data/publications/lgggbm_2016 (accessed Nov. 2026).'),
    ("esl", 'T. Hastie, R. Tibshirani and J. Friedman, The Elements of Statistical Learning, 2nd ed. New York: Springer, 2009.'),
    ("varma", 'S. Varma and R. Simon, "Bias in error estimation when using cross-validation for model selection," BMC Bioinformatics, vol. 7, art. 91, 2006.'),
    ("rf", 'L. Breiman, "Random forests," Machine Learning, vol. 45, no. 1, pp. 5-32, 2001.'),
    ("xgb", 'T. Chen and C. Guestrin, "XGBoost: a scalable tree boosting system," in Proc. 22nd ACM SIGKDD, 2016, pp. 785-794, doi: 10.1145/2939672.2939785.'),
]
REF_NUM = {k: i + 1 for i, (k, _) in enumerate(REFS)}


def cite(text):
    """Replace {key} / {key,key2} with IEEE numbers [n] / [n, m]."""
    def rep(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        if not all(k in REF_NUM for k in keys):
            return m.group(0)          # not a citation, e.g. a hyperparameter set {3, 5, 8}
        return "[" + ", ".join(str(REF_NUM[k]) for k in keys) + "]"
    return re.sub(r"\{([a-z0-9_, ]+)\}", rep, text)



# ----------------------------------------------------------------------------- one style system for the whole report
BODY = 12          # body text, bullets
H1 = 14            # section headings ("1. Introduction")
H2 = 12            # sub-headings ("1.1 Background")
SMALL = 10         # tables, captions
TEXT_W = 6.5       # usable text width in inches (8.5" page, 1" margins)
LINE = 1.15
AFTER = 6          # space after a body paragraph (pt)


class Cursor:
    """Inserts new block elements one after another, starting after a given element."""

    def __init__(self, doc, after_el):
        self.doc, self.el = doc, after_el
        self.fig_no = 0
        self.tab_no = 0

    def _add(self, new_el):
        self.el.addnext(new_el)
        self.el = new_el
        return new_el

    def para(self, text="", bold=False, italic=False, size=BODY, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
             space_before=0, space_after=AFTER, keep_next=False):
        p_el = OxmlElement("w:p")
        self._add(p_el)
        p = Paragraph(p_el, self.doc._body)
        p.alignment = align
        pf = p.paragraph_format
        pf.space_before, pf.space_after, pf.line_spacing = Pt(space_before), Pt(space_after), LINE
        pf.left_indent = pf.first_line_indent = Inches(0)
        if keep_next:
            pf.keep_with_next = True
        if text:
            self.runs(p, text, bold=bold, italic=italic, size=size)
        return p

    @staticmethod
    def runs(p, text, bold=False, italic=False, size=BODY):
        """Add text with **bold** inline markup and {cite} keys, always in the report font."""
        text = cite(text)
        for part in re.split(r"(\*\*[^*]+\*\*)", text):
            if not part:
                continue
            b = bold
            if part.startswith("**"):
                part, b = part[2:-2], True
            r = p.add_run(part)
            r.bold, r.italic = b, italic
            r.font.name, r.font.size = FONT, Pt(size)
            r._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
            r._element.rPr.rFonts.set(qn("w:cs"), FONT)

    def sub(self, text):
        return self.para(text, bold=True, size=H2, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=10,
                         space_after=4, keep_next=True)

    def bullets(self, items):
        for it in items:
            p = self.para("", align=WD_ALIGN_PARAGRAPH.LEFT, space_after=4)
            pf = p.paragraph_format
            pf.left_indent, pf.first_line_indent = Inches(0.3), Inches(-0.2)
            pf.tab_stops.add_tab_stop(Inches(0.3))
            self.runs(p, "•\t" + it)

    def figure(self, path, caption, width=6.0):
        self.fig_no += 1
        p = self.para("", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=6, space_after=4, keep_next=True)
        p.add_run().add_picture(path, width=Inches(width))
        self.para(f"Figure {self.fig_no}. {caption}", italic=True, size=SMALL,
                  align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12)
        return self.fig_no

    def table(self, rows, caption, widths):
        """Every table: caption above, full text width, 10 pt, left-aligned cells, light header row."""
        self.tab_no += 1
        self.para(f"Table {self.tab_no}. {caption}", italic=True, size=SMALL,
                  align=WD_ALIGN_PARAGRAPH.CENTER, space_before=6, space_after=4, keep_next=True)
        scale = TEXT_W / sum(widths)
        widths = [w * scale for w in widths]
        t = self.doc.add_table(rows=len(rows), cols=len(rows[0]))
        self._add(t._tbl)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        set_borders(t)
        grid = t._tbl.tblGrid
        for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
            gc.set(qn("w:w"), str(int(w * 1440)))
        for i, row in enumerate(rows):
            row_obj = t.rows[i]
            row_obj._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
            for j, val in enumerate(row):
                cell = t.cell(i, j)
                cell.width = Inches(widths[j])
                cell.text = ""
                p = cell.paragraphs[0]
                pf = p.paragraph_format
                pf.space_before, pf.space_after, pf.line_spacing = Pt(2), Pt(2), 1.0
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if val == "●" else WD_ALIGN_PARAGRAPH.LEFT
                if len(rows) <= 14 and i < len(rows) - 1:
                    pf.keep_with_next = True
                self.runs(p, str(val), bold=(i == 0), size=SMALL)
                if i == 0:
                    shade(cell, "D9D9D9")
        self.para("", size=SMALL, space_after=6)
        return self.tab_no

    def code(self, text):
        for line in text.strip("\n").split("\n"):
            p = self.para("", align=WD_ALIGN_PARAGRAPH.LEFT, space_after=0)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.left_indent = Inches(0.2)
            r = p.add_run(line if line else " ")
            p.paragraph_format.keep_with_next = True
            r.font.name, r.font.size = "Consolas", Pt(8)
            r._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
        self.para("", space_after=4)


def set_borders(table):
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single"), e.set(qn("w:sz"), "4"), e.set(qn("w:space"), "0"), e.set(qn("w:color"), "808080")
        borders.append(e)
    tbl_pr.append(borders)
    layout = OxmlElement("w:tblLayout"); layout.set(qn("w:type"), "fixed"); tbl_pr.append(layout)


def shade(cell, hex_fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"), shd.set(qn("w:color"), "auto"), shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def set_par_text(p, text):
    """Replace the text of a template paragraph, keeping the formatting of its first run."""
    runs = p.runs
    runs[0].text = text
    for r in runs[1:]:
        r._element.getparent().remove(r._element)


def clear_par(p):
    """Remove all runs/numbering/tabs/indents of a paragraph, keep it in place."""
    for child in list(p._p):
        if child.tag != qn("w:pPr"):
            p._p.remove(child)
    for tag in ("w:numPr", "w:tabs", "w:ind", "w:spacing", "w:jc", "w:contextualSpacing"):
        for n in p._p.xpath(f"./w:pPr/{tag}"):
            n.getparent().remove(n)


def add_page_numbers(doc):
    for s in doc.sections:
        p = s.footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run()
        for kind, txt in (("begin", None), (None, "PAGE"), ("end", None)):
            if kind:
                f = OxmlElement("w:fldChar"); f.set(qn("w:fldCharType"), kind); r._element.append(f)
            else:
                it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = txt; r._element.append(it)
        r.font.name, r.font.size = FONT, Pt(SMALL)


# ----------------------------------------------------------------------------- build
SECTIONS = [("intro", "1. Introduction"), ("problem", "2. Problem Statement"), ("dataset", "3. Dataset Description"),
            ("lit", "4. Literature Review"), ("method", "5. Methodology"), ("tech", "6. Technology Stack"),
            ("results", "7. Experimental Results"), ("team", "8. Team Responsibility / Self-contribution Matrix"),
            ("concl", "9. Conclusions"), ("refs", "References"), ("appendix", "Appendix: Code and Outputs")]


def build(toc_pages=None):
    from docx.enum.text import WD_TAB_ALIGNMENT
    doc = docx.Document(TEMPLATE)
    P = doc.paragraphs
    template_team_table = doc.tables[0]

    # Balanced page margins (the template had 1" left but only 0.27" right)
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(1.0)
        s.top_margin, s.bottom_margin = Inches(1.0), Inches(0.9)

    # Title page -----------------------------------------------------------
    set_par_text(P[5], TITLE)
    name_pars = [P[7], P[9], P[11], P[13]]
    for p, (name, roll) in zip(name_pars, TEAM):
        set_par_text(p, f"{name}  ({roll})")
    n5 = copy.deepcopy(P[13]._element)
    P[13]._element.addnext(n5)
    set_par_text(Paragraph(n5, P[13]._parent), f"{TEAM[4][0]}  ({TEAM[4][1]})")
    for el in [P[i]._element for i in (7, 9, 11, 13)] + [n5]:
        Paragraph(el, None).paragraph_format.space_after = Pt(4)
    for i in (8, 10, 12, 14):
        P[i]._element.getparent().remove(P[i]._element)
    # The template "centres" title-page lines with odd indents and leading spaces; centre them properly
    for i in list(range(0, 8)) + [9, 11, 13, 15, 16, 17, 18, 19]:
        p = P[i]
        pf = p.paragraph_format
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf.left_indent = pf.first_line_indent = pf.right_indent = Inches(0)
        pf.line_spacing = 1.15
        pf.space_before = Pt(0)
        for r in p.runs:
            if r._element.xpath(".//w:drawing"):
                continue
            if not r.text.strip():
                r.text = ""
            else:
                r.text = r.text.strip()
                break
    for i in (7, 9, 11, 13):
        P[i].paragraph_format.space_before = Pt(6)
    Paragraph(n5, None).paragraph_format.space_before = Pt(6)
    Paragraph(n5, None).paragraph_format.left_indent = Inches(0)
    for i in (6, 15, 16):
        P[i].paragraph_format.space_before = Pt(18)
    for r in P[5].runs:
        r.font.size = Pt(18)

    # Declaration -----------------------------------------------------------
    decl = P[21]
    for r in decl.runs:
        if "<" in r.text or ">" in r.text or "Project Title" in r.text:
            r.text = (r.text.replace("<Project Title>", TITLE).replace("Project Title>", TITLE)
                      .replace('"<', '"').replace("<", "").replace(">", ""))
    sig = [P[22], P[23], P[24], P[25]]
    s5 = copy.deepcopy(P[25]._element)
    P[25]._element.addnext(s5)
    for p, (name, roll) in zip(sig + [Paragraph(s5, P[25]._parent)], TEAM):
        set_par_text(p, f"{name} ({roll})\tSignature: ____________________")
        for ts in p._p.xpath("./w:pPr/w:tabs"):
            ts.getparent().remove(ts)
        pf = p.paragraph_format
        pf.tab_stops.add_tab_stop(Inches(3.3))
        pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(14), 1.0
    for i in (20, 21):
        P[i].paragraph_format.left_indent = P[i].paragraph_format.right_indent = Inches(0)
        P[i].paragraph_format.first_line_indent = Inches(0)
    for i in (26, 27):
        P[i].paragraph_format.left_indent = Inches(0)
        P[i].paragraph_format.space_before, P[i].paragraph_format.space_after = Pt(0), Pt(6)

    # Contents ----------------------------------------------------------------
    # remove the bracketed helper lines and the empty spacer lines between rows
    for i in (32, 34, 36, 38, 39, 41, 42, 44, 45, 47, 48, 50, 51, 53, 56):
        P[i]._element.getparent().remove(P[i]._element)
    toc_idx = {"intro": 31, "problem": 33, "dataset": 35, "lit": 37, "method": 40, "tech": 43,
               "results": 46, "team": 49, "concl": 52, "refs": 54, "appendix": 55}
    for key, title in SECTIONS:
        p = P[toc_idx[key]]
        clear_par(p)
        p.style = doc.styles["Normal"]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf = p.paragraph_format
        pf.left_indent = pf.first_line_indent = Inches(0)
        pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(14), 1.0
        pf.tab_stops.add_tab_stop(Inches(TEXT_W), WD_TAB_ALIGNMENT.RIGHT)
        page = toc_pages[key] if toc_pages else "0"
        for txt in (title.replace("Appendix: Code and Outputs", "Appendix: Code and Outputs"), "\t" + page):
            r = p.add_run(txt); r.bold = False
            r.font.name, r.font.size = FONT, Pt(BODY)
    P[30].alignment = WD_ALIGN_PARAGRAPH.RIGHT                     # "Page No."
    P[30].paragraph_format.space_after = Pt(10)

    # Body section headings ---------------------------------------------------
    H = {k: P[i]._element for k, i in dict(intro=58, problem=60, dataset=62, lit=64, method=66, tech=68,
                                           results=70, team=72, concl=76, refs=79, appendix=84).items()}
    for i in [59, 61, 63, 65, 67, 69, 71, 73, 74, 75, 77, 78, 80, 81, 82, 83, 85]:
        el = P[i]._element
        if not el.xpath(".//w:t") and not el.xpath("./w:pPr/w:sectPr"):
            el.getparent().remove(el)
    template_team_table._tbl.getparent().remove(template_team_table._tbl)   # replaced by a table in house style
    for key, title in SECTIONS:
        p = Paragraph(H[key], None)
        clear_par(p)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf = p.paragraph_format
        pf.left_indent = pf.first_line_indent = Inches(0)
        pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(8), 1.0
        pf.keep_with_next = True
        pf.page_break_before = key != "intro"
        r = p.add_run(title); r.bold = True; r.font.name, r.font.size = FONT, Pt(H1)

    c = Cursor(doc, H["intro"])
    for key, fn in (("intro", write_intro), ("problem", write_problem), ("dataset", write_dataset),
                    ("lit", write_lit), ("method", write_method), ("tech", write_tech),
                    ("results", write_results), ("team", write_team), ("concl", write_concl),
                    ("refs", write_refs), ("appendix", write_appendix)):
        c.el = H[key]
        fn(c)

    add_page_numbers(doc)
    doc.core_properties.title = TITLE
    doc.core_properties.author = ", ".join(n for n, _ in TEAM)
    doc.save(OUT)


# ----------------------------------------------------------------------------- content (plain English)
def write_intro(c):
    c.sub("Abstract")
    c.para("Doctors now sort adult brain tumours called diffuse gliomas into three types using two lab tests: "
           "one checks for a mutation in the IDH gene, the other checks whether parts of chromosomes 1 and 19 "
           "(called 1p and 19q) are missing {louis}. In this project we test whether a single measurement of "
           "gene activity in the tumour (RNA sequencing) is enough to tell the three types apart. We used 654 "
           "patients from The Cancer Genome Atlas (TCGA) {ceccarelli} and trained five models: logistic "
           "regression, a decision tree, k-nearest neighbours, a random forest and XGBoost. The models were "
           "tuned with 5-fold cross-validation on 70% of the patients, the best one was picked on a 10% "
           "validation set, and the final scores come from a 20% test set that was never used before. The "
           "chosen random forest classified 127 of 131 test patients correctly (accuracy 0.969, macro-F1 "
           "0.971). Logistic regression did slightly better (macro-F1 0.984). Eight of the ten genes the "
           "random forest relies on most sit on chromosome 1p or 19q, a pattern consistent with the known "
           "effects of 1p/19q codeletion.")
    c.sub("1.1 Background")
    c.para("Gliomas are the most common cancer that starts in the brain in adults. For a long time "
           "they were typed only by looking at tumour tissue under a microscope. Large genetic studies later "
           "showed that a few changes in the tumour's DNA say much more about how it will behave. Most "
           "slower-growing gliomas carry a mutation in the IDH1 or IDH2 gene {yan}. This mutation switches off "
           "many genes across the genome {noushmehr}. When the tumour has also lost chromosome arms 1p and 19q, "
           "it usually responds well to chemotherapy. Two TCGA studies {brat, ceccarelli} confirmed "
           "that these two changes split gliomas into clear groups, better than the microscope does.")
    c.para("The 2021 World Health Organization (WHO) classification made these changes part of the diagnosis "
           "{louis}. An adult diffuse glioma is now called one of three things: glioblastoma (no IDH mutation), "
           "astrocytoma (IDH mutation, 1p/19q present) or oligodendroglioma (IDH mutation, 1p/19q lost). The "
           "type decides the treatment and the expected outcome.")
    c.sub("1.2 Motivation")
    c.para("Finding the type needs several separate lab tests, and not every hospital can run them. RNA "
           "sequencing, on the other hand, measures the activity of about 20,000 genes in one go. Both "
           "changes leave a trace in that data: the IDH mutation switches many genes off, and losing one copy "
           "of 1p and 19q lowers the activity of genes on those arms. If a model can read these traces, one "
           "test could replace several, or at least flag cases that need a second look.")
    c.sub("1.3 Objectives")
    c.bullets([
        "Build a model that predicts the glioma type (IDH and 1p/19q status) from tumour gene activity only.",
    ])


def write_problem(c):
    c.para("**Problem statement:** Predict a glioma's molecular type (IDHwt, IDHmut-non-codel or IDHmut-codel) "
           "from tumour gene expression alone.")
    c.sub("2.1 Problem definition")
    c.para("Each patient is described by the activity of 20,330 genes measured in their tumour. Each "
           "patient also has a known type, taken from Table S1 of Ceccarelli et al. {ceccarelli}. The goal is to "
           "learn a rule that maps the gene values to the type, and that still works on new patients. This is "
           "a supervised classification problem with three classes (Table 1).")
    c.table([
        ["Class (our label)", "What it means", "WHO 2021 tumour type {louis}", "Patients"],
        ["IDHwt", "No IDH mutation", "Glioblastoma, IDH-wildtype", "229"],
        ["IDHmut-non-codel", "IDH mutation, 1p/19q present", "Astrocytoma, IDH-mutant", "256"],
        ["IDHmut-codel", "IDH mutation, 1p/19q lost", "Oligodendroglioma, IDH-mutant, 1p/19q-codeleted", "169"],
    ], "The three classes and the WHO 2021 tumour types they match.", [1.4, 1.7, 2.5, 0.8])
    c.sub("2.2 Challenges")
    c.bullets([
        "**Skewed values.** Raw gene values range from 0 to almost 900,000, and many genes are zero in most tumours.",
        "**Two similar classes.** Both IDH-mutant types share the same gene switch-off pattern {noushmehr}. They "
        "differ mainly in the 1p/19q loss.",
    ])
    c.sub("2.3 Scope")
    c.para("This is a student research project on public data. The models are not medical tools and should not "
           "be used to make decisions about patients. Only gene activity is given to the models. Age, grade and "
           "the lab results that define the label (IDH and 1p/19q status) are kept out on purpose.")


def write_dataset(c):
    c.sub("3.1 Source")
    c.para("Both files come from the official data page of the TCGA glioma study (Ceccarelli et al., Cell 2016), "
           "hosted by the US National Cancer Institute's Genomic Data Commons {ceccarelli, gdc}. We did not use "
           "copies from Kaggle or other sites, and the raw files are never changed (Table 2).")
    c.table([
        ["File", "Content", "Size", "Use"],
        ["LGG-GBM.gene_expression.normalized.txt", "Gene activity from RNA sequencing (RSEM values)",
         "667 patients × 20,330 genes", "Features"],
        ["TableS1.PatientData.20151020.v3.xlsx", "Clinical and lab details for each patient",
         "1,122 patients × 51 columns", "Label; grade for EDA"],
    ], "Source files.", [2.3, 2.0, 1.4, 1.0])
    c.sub("3.2 Joining the files")
    c.para("Table S1 lists 1,122 patients, and 974 of them have a type label. We matched these to the 667 gene "
           "profiles using the TCGA patient ID (for example TCGA-CS-4938). This gives **654 patients with "
           "20,330 genes each**. There were no missing values and no repeated patients.")
    c.sub("3.3 Features and target")
    c.table([
        ["Variable", "Data type", "Description", "Example"],
        ["Case", "Text (ID)", "TCGA patient ID, used to join the files", "TCGA-CS-4938"],
        ["20,330 gene columns", "Number, continuous, ≥ 0", "Activity of one gene; log2(x + 1) is applied before modelling", "A1BG = 49"],
        ["IDH/codel subtype (target)", "Category, 3 classes", "IDHwt, IDHmut-non-codel, IDHmut-codel; coded as 0, 1, 2", "IDHmut-codel"],
        ["Grade (EDA only)", "Category, ordered", "WHO grade G2, G3 or G4; not given to the models", "G3"],
    ], "Features, target and data types.", [1.6, 1.4, 2.6, 1.1])
    c.sub("3.4 Class balance")
    c.table([
        ["Class", "Patients", "Share", "Grade 2", "Grade 3", "Grade 4"],
        ["IDHmut-non-codel", "256", "39.1%", "113", "103", "8"],
        ["IDHwt", "229", "35.0%", "19", "66", "135"],
        ["IDHmut-codel", "169", "25.8%", "81", "70", "0"],
        ["Total", "654", "100%", "213", "239", "143"],
    ], "Patients per class and per WHO grade (grade is missing for 59 patients: 9 IDHwt, 32 IDHmut-non-codel, 18 IDHmut-codel).", [1.8, 1.0, 0.9, 1.0, 1.0, 1.0])
    c.figure(FIG + "cell13.png", "Number of patients in each class.", width=4.5)


def write_lit(c):
    c.para("The TCGA glioma data is public and well documented, so many studies have used it. Table 5 lists "
           "the work closest to ours: studies on the same TCGA data, or studies that predict the same tumour "
           "markers.")
    c.table([
        ["Study", "Data", "Task", "Method", "Main result"],
        ["Verhaak et al., 2010 {verhaak}", "TCGA glioblastoma, 202 tumours", "Glioblastoma subgroups",
         "Clustering; nearest-centroid classifier (840 genes)", "Four subgroups; one linked to IDH1"],
        ["Brat et al. (TCGA), 2015 {brat}", "TCGA lower-grade glioma, 293 tumours", "Lower-grade glioma subgroups",
         "Clustering of several data types", "IDH, 1p/19q and TP53 define 3 groups"],
        ["Ceccarelli et al., 2016 {ceccarelli}", "TCGA all gliomas, 1,122 patients (our source)", "Glioma subgroups",
         "Clustering; classifier on DNA methylation", "IDH/1p19q classes; Table S1 used here"],
        ["Capper et al., 2018 {capper}", "DNA methylation, ~2,800 brain tumours", "Brain tumour type",
         "Random forest", "Changed the diagnosis in up to 12% of cases"],
        ["Wang et al., 2019 {wang}", "TCGA RNA (692) + REMBRANDT (222)", "1p/19q loss (yes/no)",
         "GSVA, PAM, smoother and others", "98.4% (TCGA), 98.6% (REMBRANDT)"],
        ["Chen et al., 2019 {chen19}", "TCGA RNA editing, 638 tumours", "IDH; 1p/19q (yes/no)",
         "SVM, random forest, AdaBoost", "AUC 0.989 (IDH), 0.990 (1p/19q)"],
        ["Tasci et al., 2022 {tasci}", "TCGA + CGGA clinical and mutation data", "Low vs high grade",
         "Voting-based feature selection + ensemble", "87.6% (TCGA), 79.7% (CGGA)"],
        ["Munquad et al., 2022 {munquad}", "TCGA glioblastoma RNA + methylation", "Glioblastoma subgroups",
         "Convolutional neural network", "Beat classical ML models"],
        ["**This project**", "TCGA all gliomas, RNA, 654 patients", "3 WHO types",
         "LR, DT, KNN, RF, XGBoost", "Test macro-F1 0.971 (RF), 0.984 (LR)"],
    ], "Earlier work on TCGA glioma data and on predicting glioma type.", [1.4, 1.5, 1.1, 1.5, 1.5])
    c.sub("4.1 What earlier studies did")
    c.para("The first TCGA papers grouped tumours without using labels. Verhaak et al. {verhaak} found four "
           "subgroups of glioblastoma and saw that one of them often had the IDH1 mutation. The later studies "
           "{brat, ceccarelli} showed that IDH status and 1p/19q loss give the clearest split of all gliomas, "
           "and this split is the target we predict. DNA methylation, which is most directly changed by the "
           "IDH mutation, is now the basis of a widely used random-forest classifier for brain tumours {capper}.")
    c.para("Studies that trained models on the same TCGA data usually predicted IDH and 1p/19q separately, as "
           "two yes/no questions. Wang et al. {wang} predicted 1p/19q loss from gene activity with about 98% "
           "accuracy, also on a second, independent dataset. Chen et al. {chen19} reached an AUC near 0.99 for "
           "both markers using RNA editing data. Tasci et al. {tasci} predicted tumour grade from clinical and "
           "mutation data with 87.6% accuracy, and Munquad et al. {munquad} used a neural network for "
           "glioblastoma subgroups.")
    c.sub("4.2 Gap and how this project fits")
    c.para("Most earlier studies predict the two markers one at a time, use data types that are less common than "
           "RNA sequencing, or report one model without comparing it to simpler ones. We predict the three WHO "
           "types directly, use only gene activity, and compare five model types under the same fair test with "
           "confidence intervals. The numbers in Table 5 come from different tasks and test setups, so they show "
           "context rather than a ranking.")


def write_method(c):
    c.para("Table 6 shows the full pipeline. Every step that learns from the data (gene filter, scaling, PCA, "
           "model training and tuning) uses the training patients only. The gene filter, scaler and PCA are steps "
           "inside each model's scikit-learn pipeline, so during cross-validation they are re-fitted on the "
           "training part of every fold and never see that fold's held-out patients. This avoids the data leakage problem "
           "described by Varma and Simon {varma}.")
    c.table([
        ["Step", "What happens", "Data used"],
        ["1. Load and join", "Read both files and match patients by TCGA ID; drop patients without a label", "All files"],
        ["2. EDA", "Class balance, data quality, value ranges, PCA, grade table, top genes", "All 654 (only to describe)"],
        ["3. Transform", "log2(x + 1) on every value (fixed formula, learns nothing)", "All"],
        ["4. Split", "70% train / 10% validation / 20% test, same class mix in each, seed 42", "All"],
        ["5. Filter", "Keep genes active (> 0) in at least 20% of the patients the pipeline is fitted on "
         "(17,963 genes on the full training set)", "Train only, re-fitted in each CV fold"],
        ["6. Tune", "Try every setting in a grid with 5-fold cross-validation, scored by macro-F1", "Train only"],
        ["7. Select", "Pick the model with the best macro-F1 on the validation set", "Validation"],
        ["8. Evaluate", "Accuracy, macro-F1, ROC-AUC, confusion matrices, confidence intervals, top genes", "Test (used once)"],
        ["9. Demo app", "Re-train the tuned models, save them and serve them in a Streamlit app", "Train"],
    ], "The full pipeline.", [1.3, 3.8, 1.4])

    c.sub("5.1 Exploratory data analysis (EDA)")
    c.para("We explored the joined data before building any model. Table 7 lists what we found and how each "
           "finding shaped the next steps.")
    c.table([
        ["Check", "Finding", "What we did about it"],
        ["Class balance", "256 / 229 / 169 patients (39% / 35% / 26%)", "Kept the class mix in every split; used macro-F1"],
        ["Missing values", "None; no repeated patients", "No filling-in needed"],
        ["Size", "20,330 genes but only 654 patients", "Strong regularisation; simple baseline"],
        ["Empty genes", "183 genes are zero in every patient; 1,879 are zero in over 90%", "Filter out rarely active genes"],
        ["Skew", "Values up to 884,605; median 519", "log2(x + 1) transform"],
        ["PCA", "IDHwt stands apart; the two IDH-mutant classes overlap", "Expect most mistakes between IDH-mutant classes"],
        ["Grade", "IDHwt is mostly grade 4; IDHmut-codel is never grade 4", "Grade kept out of the model; noted as a limitation"],
        ["Top genes", "Clear block pattern; IDHwt high in CHI3L1, PDPN, EMP3", "Strong signal that makes biological sense"],
    ], "Summary of the exploratory data analysis.", [1.3, 2.8, 2.6])
    c.figure(FIG + "cell17.png", "All gene values before (left, cut at the 99th percentile) and after the "
             "log2(x + 1) transform (right).", width=6.0)
    c.para("Raw gene values are very uneven: a few genes have huge values and most are small (Figure 2). After "
           "the log2(x + 1) transform the values are spread much more evenly, apart from a peak at zero for "
           "inactive genes. This transform is standard for RNA sequencing data {esl}.")
    c.figure(FIG + "cell19.png", "The first two principal components of the log-transformed data, coloured by class.",
             width=4.5)
    c.para("Principal component analysis (PCA) squeezes the 20,000 genes into a few summary directions. In "
           "Figure 3, IDHwt tumours form their own cloud, which fits the large gene switch-off caused by the IDH "
           "mutation {noushmehr}. The two IDH-mutant classes overlap, so telling them apart needs information "
           "spread over many genes, mainly the lower activity of genes on 1p and 19q.")
    c.para("Table 4 shows that grade and type are "
           "linked: IDHwt tumours are mostly grade 4, and IDHmut-codel tumours never are. A model could partly "
           "learn tumour severity instead of type. We come back to this in Section 9.")

    c.sub("5.2 Data preparation and feature engineering")
    c.bullets([
        "**log2(x + 1).** Applied to every value. It is a fixed formula, so it cannot leak test information.",
        "**Split first.** 20% of patients go to the test set, then 10% of all patients go to the validation set. "
        "Both keep the same class mix (Table 8).",
        "**Gene filter (first pipeline step).** Genes active in fewer than 20% of the patients the pipeline is "
        "fitted on are removed. It is written as a small scikit-learn transformer and placed first in every "
        "model's pipeline, so each cross-validation fold learns its own gene list from its training part only. "
        "Fitted on all 457 training patients, it keeps 17,963 of the 20,330 genes, and that same list is "
        "then applied to the validation and test patients.",
        "**Scaling.** Logistic regression and KNN need genes on the same scale, so each gene is standardised. "
        "The scaler sits inside a scikit-learn pipeline, so in cross-validation it only sees the training part "
        "of each fold.",
        "**PCA for KNN.** Distances stop being useful with thousands of genes, so KNN works on 10 or 30 "
        "principal components {esl}.",
        "**Labels as numbers.** IDHwt = 0, IDHmut-non-codel = 1, IDHmut-codel = 2.",
    ])
    c.table([
        ["Class", "Train", "Validation", "Test"],
        ["IDHmut-non-codel", "179", "26", "51"],
        ["IDHwt", "160", "23", "46"],
        ["IDHmut-codel", "118", "17", "34"],
        ["Total", "457 (70%)", "66 (10%)", "131 (20%)"],
    ], "Patients per class in the three sets.", [2.0, 1.5, 1.5, 1.5])

    c.sub("5.3 Baseline model: logistic regression")
    c.para("Our baseline is multinomial logistic regression with an L2 (ridge) penalty. It is the usual "
           "first choice for gene data for three reasons: it is simple, it is hard to overfit once "
           "penalised, and its weights show how much each gene matters. The setting C controls the penalty: a smaller C pushes all gene weights closer to zero "
           "{esl}. We tried C = 0.0001, 0.001, 0.01, 0.1 and 1.")
    c.sub("5.4 Models used for comparison")
    c.bullets([
        "**Decision tree**. A set of yes/no rules on single genes. It is easy to read and it is the "
        "building block of both ensembles, so it shows how much the ensembles add. We limited its depth "
        "(3, 5, 8 or no limit), the smallest leaf size (1, 5 or 15) and the pruning strength (0, 0.005 or 0.02).",
        "**k-nearest neighbours (KNN)**. Gives a new tumour the most common type among its closest "
        "training tumours. It tests whether similar tumours share a type. We tried 10 or 30 PCA components, "
        "k = 3, 7, 15 or 25 neighbours, and equal or distance-based voting.",
    ])
    c.sub("5.5 Ensemble models")
    c.bullets([
        "**Random forest (bagging)** {rf}. Builds many trees, each on a random sample of patients and a random "
        "subset of genes, and lets them vote. Averaging many trees makes the result more stable than a single "
        "tree. We tried 200 or 500 trees, √p or 20% of genes per split, and a smallest leaf of 1, 3 or 8.",
        "**XGBoost (boosting)** {xgb}. Builds small trees one after another, where each new tree fixes the "
        "mistakes of the trees before it. A small learning rate, shallow trees, an L2 penalty and random "
        "sampling of patients (80%) and genes keep it from overfitting. We tried 100 or 300 trees, depth 2 or "
        "4, learning rate 0.05 or 0.2, 20% or 50% of genes per tree, and penalty λ = 1 or 10.",
    ])
    c.sub("5.6 Tuning and model selection")
    c.para("For each model we tried every setting in its grid using 5-fold stratified cross-validation on the 457 "
           "training patients, scored by macro-F1. The best setting was re-trained on all training "
           "patients. The five tuned models were then compared on the 66 validation patients, and the one with "
           "the best validation macro-F1 became the final model. The 131 test patients were used only once, "
           "after every choice had been made.")
    c.sub("5.7 Evaluation measures")
    c.para("Because the classes have different sizes, plain accuracy is not enough. **Macro-F1** is the "
           "main measure: it averages the F1 score of each class, so every class counts equally. We also report "
           "accuracy, balanced accuracy and ROC-AUC (how well the predicted probabilities rank the classes). The "
           "**overfit gap** is training accuracy minus test accuracy. To show how certain the test scores are, we "
           "resampled the test set 1,000 times and took the middle 95% of the macro-F1 values as a confidence "
           "interval.")


def write_tech(c):
    c.table([
        ["Area", "Tool", "Used for"],
        ["Language", "Python 3", "All analysis and the app"],
        ["Data handling", "pandas, NumPy, openpyxl", "Reading the files, joining, number arrays"],
        ["Machine learning", "scikit-learn, XGBoost {xgb}", "Pipelines, scaling, PCA, LR, decision tree, KNN, random forest, XGBoost, grid search, metrics"],
        ["Charts", "Matplotlib", "EDA plots, confusion matrices, gene importance"],
        ["Notebook", "Jupyter", "glioma_analysis.ipynb, the full analysis"],
        ["Saving models", "joblib", "Storing the five tuned models for the app"],
        ["Web app", "Streamlit", "Interactive demo (app/app.py)"],
    ], "Technology stack.", [1.4, 2.1, 3.0])


def write_results(c):
    c.sub("7.1 Cross-validation and regularisation")
    c.table([
        ["Model", "Best settings", "5-fold CV macro-F1"],
        ["Logistic Regression", "C = 0.001", "0.983"],
        ["Decision Tree", "max depth 3, smallest leaf 15, no pruning", "0.931"],
        ["KNN", "10 PCA components, k = 3, distance-based voting", "0.953"],
        ["Random Forest", "200 trees, 20% of genes per split, smallest leaf 8", "0.983"],
        ["XGBoost", "300 trees, depth 2, learning rate 0.05, 20% of genes per tree, λ = 1", "0.989"],
    ], "Best settings and cross-validated macro-F1 on the training set.", [1.6, 3.6, 1.3])
    c.para("Every model chose strong regularisation. Logistic regression picked C = 0.001, a heavy penalty: with "
           "almost 18,000 genes, many small weights work better than a few large ones. The decision tree picked "
           "a depth of only 3 with at least 15 patients per leaf. XGBoost picked its most careful settings "
           "(depth 2, learning rate 0.05, 20% of genes per tree), and the random forest preferred larger leaves.")
    c.sub("7.2 Choosing the final model")
    c.table([
        ["Model", "CV macro-F1", "Validation accuracy", "Validation macro-F1"],
        ["Random Forest", "0.983", "1.000", "1.000"],
        ["Logistic Regression", "0.983", "0.985", "0.986"],
        ["XGBoost", "0.989", "0.985", "0.986"],
        ["Decision Tree", "0.931", "0.924", "0.928"],
        ["KNN", "0.953", "0.909", "0.915"],
    ], "Results on the 66 validation patients. The random forest is chosen.", [1.9, 1.5, 1.6, 1.5])
    c.para("The random forest got all 66 validation patients right, so it was chosen. Logistic regression and "
           "XGBoost each made one mistake. With only 66 patients, one mistake changes macro-F1 by about 1.5 points, "
           "so the top three models are really tied at this stage.")
    c.sub("7.3 Results on the test set")
    c.table([
        ["Model", "Accuracy", "Bal. accuracy", "Macro-F1", "ROC-AUC", "Overfit gap"],
        ["Logistic Regression", "0.985", "0.987", "0.984", "0.992", "0.015"],
        ["**Random Forest (chosen)**", "0.969", "0.972", "0.971", "0.997", "0.028"],
        ["XGBoost", "0.969", "0.972", "0.970", "0.999", "0.031"],
        ["Decision Tree", "0.947", "0.952", "0.949", "0.956", "0.012"],
        ["KNN", "0.931", "0.935", "0.934", "0.972", "0.069"],
    ], "Results on the 131 test patients (overfit gap = training accuracy − test accuracy).",
        [2.1, 0.9, 1.1, 0.9, 0.9, 1.0])
    c.figure(FIG + "test_f1_ci.png", "Test accuracy (grey) and macro-F1 (blue) with 95% bootstrap confidence intervals (1,000 resamples).", width=6.0)
    c.para("On the 131 unseen test patients, the chosen random forest reached **0.969 accuracy and 0.971 "
           "macro-F1** (95% confidence interval 0.939 to 0.994), with ROC-AUC 0.997. Logistic regression scored "
           "highest on the test set (0.984), but it was not the validation pick, and we did not change our choice "
           "after seeing the test results. The confidence intervals of the top three models overlap almost "
           "completely (Figure 4). On 131 patients one mistake is worth about 0.8 accuracy points, so the data "
           "cannot say which of the three is truly best.")
    c.figure(FIG + "cv_val_test.png", "Macro-F1 of each model in cross-validation, on the validation set and on "
             "the test set.", width=6.0)
    c.para("For the three strong models, cross-validation and test scores are within about 1.5 points of each "
           "other (Figure 5), so the cross-validation results were reliable. The random forest's perfect "
           "validation score was partly luck, since its test score dropped back to its cross-validation level. "
           "KNN has the biggest overfit gap (0.069).")
    c.sub("7.4 Where the model makes mistakes")
    c.figure(FIG + "cell55.png", "Confusion matrices of the five models on the test set (rows: true class, "
             "columns: predicted class).", width=6.0)
    c.table([
        ["Class", "Precision", "Recall", "F1", "Patients"],
        ["IDHwt", "0.98", "0.96", "0.97", "46"],
        ["IDHmut-non-codel", "0.96", "0.96", "0.96", "51"],
        ["IDHmut-codel", "0.97", "1.00", "0.99", "34"],
        ["Macro average", "0.97", "0.97", "0.97", "131"],
    ], "Results per class for the chosen random forest.", [2.1, 1.1, 1.1, 1.1, 1.1])
    c.table([
        ["Patient", "Grade", "True class", "Predicted class"],
        ["TCGA-26-5133", "G4", "IDHwt", "IDHmut-non-codel"],
        ["TCGA-FG-8181", "G3", "IDHwt", "IDHmut-non-codel"],
        ["TCGA-06-0178", "G4", "IDHmut-non-codel", "IDHwt"],
        ["TCGA-CS-5394", "G3", "IDHmut-non-codel", "IDHmut-codel"],
    ], "The four test patients the random forest got wrong (from the demo app).", [1.6, 0.9, 2.0, 2.0])
    c.para("The random forest made 4 mistakes out of 131 (Tables 13 and 14). It never missed an IDHmut-codel "
           "tumour. Three of the four mistakes are between IDHwt and IDHmut-non-codel, and they happen exactly "
           "where grade is misleading: an IDH-mutant grade 4 tumour was called IDHwt, and two IDHwt tumours of "
           "grade 3 and 4 were called IDH-mutant. This supports the EDA finding that grade and type are linked.")
    c.sub("7.5 Ensembles compared with single models")
    c.bullets([
        "**Random forest vs a single tree.** The forest raised test macro-F1 from 0.949 to 0.971 and ROC-AUC from "
        "0.956 to 0.997. Averaging many trees makes the result more stable "
        "{rf}.",
        "**XGBoost vs random forest.** They are tied on correct predictions (0.970 vs 0.971 macro-F1). "
        "Boosting did not beat bagging on accuracy here.",
        "**Linear vs tree-based models.** The penalised logistic regression did as well as, or better than, both "
        "ensembles. With many genes and few patients, a simple penalised model is very hard to beat {esl}.",
    ])
    c.sub("7.6 Which genes the model uses")
    c.figure(FIG + "cell61.png", "The 10 most important genes in the random forest.", width=4.5)
    c.table([
        ["Gene", "Location", "Note"],
        ["RBP1", "3q23", "Switched off in almost all IDH-mutant gliomas {chou}"],
        ["KIAA0495", "1p36.32", "On chromosome arm 1p"],
        ["PSMB2", "1p34.3", "On chromosome arm 1p"],
        ["MIER1", "1p31.3", "On chromosome arm 1p"],
        ["FBXO42", "1p36.13", "On chromosome arm 1p"],
        ["CSDE1", "1p13.2", "On chromosome arm 1p"],
        ["FBXO17", "19q13.2", "On chromosome arm 19q"],
        ["TMEM167B", "1p13.3", "On chromosome arm 1p"],
        ["FKBP9", "7p14.3", "Chromosome 7, often gained in glioblastoma {louis}"],
        ["EMP3", "19q13.33", "On chromosome arm 19q; high in IDHwt"],
    ], "Location of the 10 most important genes (from NCBI Gene).", [1.6, 1.2, 3.7])
    c.para("The gene list makes biological sense. The top gene, RBP1, is known to be switched off in almost all "
           "IDH-mutant gliomas {chou}, so it is a natural marker of IDH status. **Eight of the ten genes sit on "
           "chromosome arm 1p (six) or 19q (two).** Tumours that have lost 1p and 19q keep only one copy of these "
           "genes, so their activity drops. The importance pattern is therefore consistent with the "
           "biological effects of 1p/19q codeletion. It is a plausibility check, not independent proof that the "
           "model has detected the codeletion itself.")
    c.sub("7.7 Demo application")
    c.para("We saved the tuned models and built a Streamlit app (Figures 8 and 9). The user picks a "
           "test patient or uploads a file of gene values. The app applies the same log transform and gene list, "
           "and shows the predicted type, the matching WHO type and the probability of each class. Its "
           "whole-test-set view gives the same confusion matrix as the notebook.")
    c.figure(FIG + "demo_single.png", "Demo: prediction for test patient TCGA-06-0184 (grade 4), correctly predicted "
             "as IDHwt with 88.5% confidence.", width=6.0)
    c.figure(FIG + "demo_testset_crop.png", "Demo: confusion matrix and wrongly classified patients on the whole "
             "test set.", width=6.0)


def write_team(c):
    c.para("Table 16 shows what each group member contributed to each activity.")
    # in the order the work was done; columns: Raghav, Darshita, Prerit, Saksham, Rayna
    matrix = [
        ("Literature review",                        (0, 1, 0, 0, 1)),
        ("Data preprocessing & EDA",                 (1, 1, 0, 1, 0)),
        ("Feature engineering",                      (1, 0, 1, 1, 0)),
        ("Model building",                           (1, 1, 1, 1, 0)),
        ("Ensemble",                                 (0, 0, 1, 0, 1)),
        ("Hyperparameter tuning / cross-validation", (0, 0, 0, 0, 1)),
        ("Model comparisons",                        (0, 1, 0, 1, 1)),
        ("Model evaluation and testing",             (1, 0, 1, 0, 0)),
        ("Interpreting results",                     (0, 1, 0, 0, 1)),
        ("Demo",                                     (1, 0, 0, 0, 0)),
        ("Report",                                   (1, 0, 0, 0, 0)),
        ("Slides",                                   (1, 1, 1, 1, 1)),
    ]
    c.table([["Activity"] + [f"{n} ({r})" for n, r in TEAM]] +
            [[a] + ["●" if m else "" for m in marks] for a, marks in matrix],
            "Team responsibility and self-contribution matrix.", [2.0, 0.9, 0.9, 0.9, 0.9, 0.9])


def write_concl(c):
    c.sub("9.1 Summary")
    c.para("This project shows that tumour gene activity alone can tell the three WHO glioma types apart with "
           "high accuracy. On 131 patients that were never used for training, tuning or model choice, the chosen "
           "random forest got 127 right (accuracy 0.969, macro-F1 0.971, 95% confidence interval 0.939 to 0.994). "
           "The penalised logistic regression did slightly better, with macro-F1 0.984. The main findings are:")
    c.bullets([
        "Gene activity alone separates the three types, with test macro-F1 between 0.93 (KNN) and 0.98 "
        "(logistic regression). The chosen model never missed an IDHmut-codel tumour.",
        "The random forest clearly improved on a single decision tree (macro-F1 0.971 vs 0.949). XGBoost tied "
        "with the random forest.",
        "With many genes and few patients, a simple penalised linear model works as well as any ensemble and "
        "is easier to explain.",
        "The model relies on real biology: RBP1, a gene switched off by the IDH mutation, and eight genes on "
        "chromosome arms 1p and 19q, which are lost in oligodendroglioma.",
    ])
    c.sub("9.2 Limitations")
    c.bullets([
        "**One dataset only.** All patients come from TCGA. We have not tested the model on data from other "
        "hospitals, which may differ because of lab and equipment effects.",
        "**Small test sets.** The validation set (66 patients) cannot reliably separate models that differ by "
        "one patient, and one 131-patient test set gives wide confidence intervals.",
        "**Grade is linked to type.** IDHwt tumours are mostly grade 4 and IDHmut-codel tumours never are. "
        "Three of the four test mistakes are tumours where this link breaks.",
        "**Older labels.** The TCGA labels come from before the 2021 WHO rules, so a few IDHwt lower-grade "
        "tumours might be typed differently today.",
        "**Not a medical tool.** The work is a student research project and has not been tested in a clinic.",
    ])
    c.sub("9.3 Future work")
    c.bullets([
        "Test the TCGA-trained model on another dataset such as CGGA, after matching gene names and "
        "correcting batch effects.",
        "Use repeated cross-validation instead of one fixed split {varma}.",
        "Check accuracy separately within each grade.",
        "Put the demo app online on a cloud platform.",
    ])


def write_refs(c):
    for i, (_, text) in enumerate(REFS, 1):
        p = c.para("", align=WD_ALIGN_PARAGRAPH.LEFT, space_after=4)
        p.paragraph_format.left_indent = Inches(0.45)
        p.paragraph_format.first_line_indent = Inches(-0.45)
        p.paragraph_format.tab_stops.add_tab_stop(Inches(0.45))
        Cursor.runs(p, f"[{i}]\t{text}", size=SMALL)


def write_appendix(c):
    c.sub("A. Key code from glioma_analysis.ipynb")
    c.para("The full notebook, with all outputs, is submitted with this report. The code below shows the data "
           "split, the gene filter as a pipeline step and the model search.")
    c.code('''expr = pd.read_csv(RAW + "LGG-GBM.gene_expression.normalized.txt",
                   sep="\\t", index_col=0)
meta = pd.read_excel(RAW + "TableS1.PatientData.20151020.v3.xlsx",
                     sheet_name=0, header=1).set_index("Case")
labels = meta["IDH/codel subtype"].dropna()
ids = expr.index.intersection(labels.index)
X, y = expr.loc[ids], labels.loc[ids]                  # 654 patients x 20,330 genes

X_log = np.log2(X + 1)
X_rest, X_te, y_rest, y_te = train_test_split(
    X_log, y, test_size=0.20, stratify=y, random_state=42)
X_tr, X_va, y_tr, y_va = train_test_split(
    X_rest, y_rest, test_size=0.125, stratify=y_rest, random_state=42)
class LowExpressionFilter(BaseEstimator, TransformerMixin):
    """Keep genes expressed (> 0) in >= min_fraction of the patients seen in fit()."""
    def __init__(self, min_fraction=0.2):
        self.min_fraction = min_fraction
    def fit(self, X, y=None):
        self.keep_ = (np.asarray(X) > 0).mean(axis=0) >= self.min_fraction
        return self
    def transform(self, X):
        return X.loc[:, self.keep_]''')
    c.code('''models["Logistic Regression"] = (
    Pipeline([("filter", LowExpressionFilter(0.2)),    # re-fitted in every CV fold
              ("scale", StandardScaler()),
              ("clf", LogisticRegression(max_iter=500, random_state=42))]),
    {"clf__C": [0.0001, 0.001, 0.01, 0.1, 1]})
# Decision Tree, KNN (scaler + PCA), Random Forest and XGBoost also start with the filter

cv = StratifiedKFold(5, shuffle=True, random_state=42)
for name, (pipeline, grid) in models.items():
    search = GridSearchCV(pipeline, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
    search.fit(X_tr, y_tr)
    best[name] = search.best_estimator_

selected = val_table["Val macro-F1"].idxmax()   # choice uses the validation set only''')
    c.sub("B. Notebook output: summary of the run")
    c.code('''- Patients: 654 total -> 457 train / 66 validation / 131 test
- Selected model (best validation macro-F1): Random Forest
- Its test scores: accuracy 0.969, macro-F1 0.971 (95% CI 0.939-0.994),
  ROC-AUC 0.997
- Its overfit gap (train - test accuracy): 0.028
- Bagging vs single tree (Random Forest vs Decision Tree): 0.971 vs 0.949
- Boosting vs bagging (XGBoost vs Random Forest): 0.970 vs 0.971
- Largest overfit gap: KNN (0.069); lowest test macro-F1: KNN (0.934)
- Most frequent error: IDHwt predicted as IDHmut-non-codel (2 patients)''')
    c.sub("C. Project files")
    c.bullets([
        "glioma_analysis.ipynb: the full analysis with outputs",
        "app/train_models.py and app/app.py: the demo app",
        "report/make_figures.py, report/build_report.py, report/build_slides.py: scripts that create the charts, "
        "this report and the slides",
        "data/raw/: the two source files from the NCI Genomic Data Commons {gdc}",
    ])


def find_pages(pdf_path):
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    start = {}
    for i, page in enumerate(reader.pages):
        lines = [l.strip() for l in (page.extract_text() or "").split("\n")][:4]
        if i < 3 or any(l.startswith("Contents") for l in lines):
            continue
        for k, title in SECTIONS:
            if k not in start and any(l.startswith(title[:18]) for l in lines):
                start[k] = i + 1
    order = [k for k, _ in SECTIONS]
    out = {}
    for j, k in enumerate(order):
        end = (start[order[j + 1]] - 1) if j + 1 < len(order) else len(reader.pages)
        out[k] = str(start[k]) if end <= start[k] else f"{start[k]}-{end}"
    return out


if __name__ == "__main__":
    from docx2pdf import convert
    pdf = OUT.replace(".docx", ".pdf")
    build(None); convert(OUT, pdf)
    pages = find_pages(pdf); print("pages:", pages)
    build(pages); convert(OUT, pdf)
    assert find_pages(pdf) == pages, "page numbers shifted"
    print("written", OUT, "and", pdf)
