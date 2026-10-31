# -*- coding: utf-8 -*-
with open('main.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace Group 5 multirow with direct makecell
old_g5 = r"\multirow{1}{*}{\makecell[l]{\textbf{Group 5:}\\\textbf{Mutual Info}}}"
new_g5 = r"\makecell[l]{\scriptsize\textbf{Group 5:}\\\scriptsize\textbf{Mutual Info}}"

# Replace other groups with scriptsize makecells
old_g1 = r"\multirow{4}{*}{\makecell[l]{\textbf{Group 1:}\\\textbf{Empirical Risk}}}"
new_g1 = r"\multirow{4}{*}{\makecell[l]{\scriptsize\textbf{Group 1:}\\\scriptsize\textbf{Empirical Risk}}}"

old_g2 = r"\multirow{2}{*}{\makecell[l]{\textbf{Group 2:}\\\textbf{Regularization}}}"
new_g2 = r"\multirow{2}{*}{\makecell[l]{\scriptsize\textbf{Group 2:}\\\scriptsize\textbf{Regularization}}}"

old_g3 = r"\multirow{2}{*}{\makecell[l]{\textbf{Group 3:}\\\textbf{Group Robust}}}"
new_g3 = r"\multirow{2}{*}{\makecell[l]{\scriptsize\textbf{Group 3:}\\\scriptsize\textbf{Group Robust}}}"

old_g4 = r"\multirow{4}{*}{\makecell[l]{\textbf{Group 4:}\\\textbf{Adversarial}}}"
new_g4 = r"\multirow{4}{*}{\makecell[l]{\scriptsize\textbf{Group 4:}\\\scriptsize\textbf{Adversarial}}}"

old_g6 = r"\multirow{2}{*}{\makecell[l]{\textbf{Group 6:}\\\textbf{Proposed Full}}}"
new_g6 = r"\multirow{2}{*}{\makecell[l]{\scriptsize\textbf{Group 6:}\\\scriptsize\textbf{Proposed Full}}}"

text = text.replace(old_g1, new_g1)
text = text.replace(old_g2, new_g2)
text = text.replace(old_g3, new_g3)
text = text.replace(old_g4, new_g4)
text = text.replace(old_g5, new_g5)
text = text.replace(old_g6, new_g6)

with open('main.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print("Applied fix_table3 replacements!")
