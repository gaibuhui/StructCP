# -*- coding: utf-8 -*-
"""
Fix Appendix-prefix duplication in paper_kbs.tex:
elsarticle's \\ref on appendix sections already emits "Appendix X",
so hard-coded "Appendix~\\ref{...}" becomes "Appendix Appendix X".

Rules:
  1) (Appendix~\\ref{sec:exch_failure}) is a MAIN-text section mislabeled as
     an appendix in the original ICLR draft -> Sec.~\\ref{...}
  2) "Appendices~\\ref{app:exch_metrics} and~\\ref{app:assumption_h}"
     -> "\\ref{...} and~\\ref{...}"  (renders "Appendix F and Appendix G")
  3) "Appendix~\\ref{X}" / "App.~\\ref{X}" -> "\\ref{X}" (prefix auto-added)
  4) "Sec.~\\ref{X}" where X is an appendix label (e.g. in appendix L
     referencing appendix I) -> "\\ref{X}"
"""
import io

PATH = r"<REPO_ROOT>/docs/paper_kbs.tex"
s = io.open(PATH, encoding="utf-8").read()
orig = s

# 1) main-text section that the draft mislabeled as an appendix
s = s.replace("Appendix~\\ref{sec:exch_failure}", "Sec.~\\ref{sec:exch_failure}")

# 2) plural "Appendices~..." phrasing
s = s.replace(
    "Appendices~\\ref{app:exch_metrics} and~\\ref{app:assumption_h}",
    "\\ref{app:exch_metrics} and~\\ref{app:assumption_h}",
)

# 3) generic prefix removal (elsarticle adds "Appendix " itself)
s = s.replace("Appendix~\\ref{", "\\ref{")
s = s.replace("App.~\\ref{", "\\ref{")

# 4) "Sec.~\\ref{<appendix-label>}" -> "\\ref{<appendix-label>}"
for lbl in [
    "sec:filter",
    "sec:robust_calib",
    "sec:community_mit",
    "sec:efficiency",
    "sec:limitations",
    "sec:appendix_h1",
]:
    s = s.replace("Sec.~\\ref{" + lbl + "}", "\\ref{" + lbl + "}")

io.open(PATH, "w", encoding="utf-8").write(s)

# report changes
import difflib
d = list(difflib.unified_diff(orig.splitlines(), s.splitlines(), lineterm=""))
print("changed lines:", sum(1 for l in d if l.startswith("+") and not l.startswith("+++")))
for l in d:
    if l.startswith("+") and not l.startswith("+++"):
        print(l[:160])
