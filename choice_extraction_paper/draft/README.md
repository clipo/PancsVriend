# Draft (LaTeX)

Build the PDF:

```
cd choice_extraction_paper/draft
latexmk -pdf main.tex
```

- `main.tex` pulls in `sections/*.tex` and `references.bib`.
- For the official ACL layout, copy `acl.sty` and `acl_natbib.bst` from https://github.com/acl-org/acl-style-files into this folder. `main.tex` uses them automatically when they are present, and a plain layout otherwise.
- `\tbd{...}` prints red placeholders for numbers that wait on experiments (see `../todo_experiments.md`).
- Works on Overleaf too: upload this folder and set `main.tex` as the main file.
