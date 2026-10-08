# -*- coding: utf-8 -*-
"""
Convert paper_iclr.tex (ICLR 2027 format) -> paper_kbs.tex (Elsevier/KBS format).

Strategy: keep the body (Introduction..Conclusion) and appendix verbatim;
replace preamble + frontmatter (title/authors/abstract) + add keywords/highlights;
switch bibliography to elsarticle-num style.
"""
import io, sys

SRC = r"<REPO_ROOT>/docs/paper_iclr.tex"
DST = r"<REPO_ROOT>/docs/paper_kbs.tex"

src = io.open(SRC, encoding="utf-8").read()

# ---------- anchors ----------
marker_doc = "\\begin{document}"
marker_intro = "\\section{Introduction}"
marker_refs = "{\\centering\\bfseries\\Large References\\par}"
marker_appendix = "\\appendix"

idx_doc = src.index(marker_doc)
idx_intro = src.index(marker_intro, idx_doc)
idx_refs = src.index(marker_refs, idx_intro)
idx_appendix = src.index(marker_appendix, idx_refs)

# ---------- abstract text ----------
front_old = src[idx_doc:idx_intro]
abs_start = front_old.index("\\begin{abstract}") + len("\\begin{abstract}")
abs_end = front_old.index("\\end{abstract}")
abstract_text = front_old[abs_start:abs_end].strip()

# ---------- new preamble ----------
new_preamble = r"""\documentclass[preprint,12pt]{elsarticle}

%% =====================================================================
%% KBS (Knowledge-Based Systems, Elsevier) submission version
%% Converted from the ICLR 2027 format (paper_iclr.tex), body & appendix
%% kept verbatim.  Only preamble / frontmatter / bibliography style change.
%% =====================================================================

% Math macros shared with the ICLR version (no clash with elsarticle)
\input{math_commands}

\usepackage{amssymb}
\usepackage{booktabs}       % professional-quality tables
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{nicefrac}       % compact 1/2, 1/6 fractions in tables
\usepackage{makecell}
\usepackage{multirow}
\usepackage{enumitem}
\usepackage[numbers]{natbib}
\usepackage{url}
% hyperref loaded last (elsarticle-compatible)
\usepackage[hidelinks,breaklinks]{hyperref}

% compress vertical spacing for a compact layout
\setlength{\belowdisplayskip}{4pt plus 1pt minus 1pt}
\setlength{\belowdisplayshortskip}{3pt plus 1pt minus 1pt}
\setlength{\abovedisplayskip}{4pt plus 1pt minus 1pt}
\setlength{\abovedisplayshortskip}{3pt plus 1pt minus 1pt}
\setlist[itemize]{topsep=2pt,itemsep=1pt,parsep=1pt}
\setlist[enumerate]{topsep=2pt,itemsep=1pt,parsep=1pt}

"""
# ---------- new frontmatter ----------
new_front = r"""\begin{document}

\begin{frontmatter}

\title{StructCP: Structure-Aware Empirical Calibration for Graph Anomaly Detection under Normality Shift}

\author[structcp]{Anonymous Author\corref{cor1}}
\cortext[cor1]{Corresponding author.}
\affiliation[structcp]{organization={Anonymous Institution},country={}}

\begin{abstract}
""" + abstract_text + r"""
\end{abstract}

\begin{keyword}
graph anomaly detection \sep normality shift \sep conformal prediction \sep test-time adaptation \sep weighted recalibration \sep pseudo-normal augmentation
\end{keyword}

\begin{highlights}
\item Drop-in, optimization-free wrapper recalibrating any frozen GAD detector
\item One $k$-NN graph reused for both test-time propagation and consistency weighting
\item Holds FPR$\le0.05$ on community-type graphs across four benchmarks and five seeds
\item Outperforms split conformal, density-ratio, and calibration baselines on four graphs
\item Honest scope boundary: weak backbones and large-$\alpha$ regimes
\end{highlights}

\end{frontmatter}

"""

# ---------- new bibliography block ----------
new_refs = r"""\bibliographystyle{elsarticle-num}
\bibliography{references}

"""

# ---------- assemble ----------
result = (new_preamble
          + new_front
          + src[idx_intro:idx_refs]      # body verbatim
          + new_refs
          + src[idx_appendix:])          # appendix verbatim

io.open(DST, "w", encoding="utf-8").write(result)
print("written:", DST)
print("chars:", len(result))
