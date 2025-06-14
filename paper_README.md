# Scientific Paper: LLM Agents in Schelling Model

This directory contains a Quarto document prepared for submission to Advances in Complex Systems or similar journals.

## Files

- `schelling_llm_paper.qmd` - Main Quarto document with embedded R code
- `references.bib` - Bibliography file
- `convergence_analysis_detailed.csv` - Convergence statistics data
- `pairwise_comparison_results.csv` - Pairwise comparison statistics

## Requirements

To render this paper, you need:

1. **Quarto** (https://quarto.org/)
   ```bash
   # macOS
   brew install quarto
   
   # Or download from https://quarto.org/docs/get-started/
   ```

2. **R** with required packages:
   ```r
   install.packages(c("tidyverse", "ggplot2", "kableExtra", 
                      "patchwork", "broom", "effsize"))
   ```

3. **LaTeX** (for PDF output)
   - macOS: `brew install --cask mactex`
   - Or use TinyTeX: `quarto install tinytex`

## Rendering the Paper

1. **PDF Output** (recommended for journal submission):
   ```bash
   quarto render schelling_llm_paper.qmd --to pdf
   ```

2. **HTML Output** (for web viewing):
   ```bash
   quarto render schelling_llm_paper.qmd --to html
   ```

3. **Word Output** (for collaboration):
   ```bash
   quarto render schelling_llm_paper.qmd --to docx
   ```

## Customization for Journal Submission

### For Advances in Complex Systems

1. Update the YAML header with journal-specific formatting:
   ```yaml
   format:
     pdf:
       documentclass: article
       classoption: [twocolumn]  # If required
       geometry: margin=1in
   ```

2. Add journal-specific LaTeX packages if needed:
   ```yaml
   header-includes:
     - \usepackage{acs-style}  # If journal provides style file
   ```

3. Adjust abstract length to journal requirements (typically 150-250 words)

### For Other Journals

- **JASSS**: Use their LaTeX template
- **Complexity**: Follow Wiley guidelines
- **PLOS ONE**: Use their template

## Key Sections to Update

1. **Author Information**: Add your name, affiliation, and email
2. **Abstract**: Currently 198 words, adjust as needed
3. **Keywords**: Add/modify based on journal requirements
4. **Data Availability**: Add actual repository URL
5. **References**: Add any additional citations

## Figure and Table Guidelines

- All figures are generated from data using R code
- Tables use `kableExtra` for professional formatting
- Captions are included for all figures and tables
- Color scheme is colorblind-friendly

## Reproducibility

The R code chunks in the document:
- Load and process the experimental data
- Generate all figures and tables
- Perform statistical analyses
- Are fully reproducible with the provided CSV files

## Tips for Submission

1. Check journal guidelines for:
   - Word count limits
   - Figure number limits
   - Reference formatting style
   - Supplementary material policies

2. Consider moving detailed statistics to supplementary material if space-constrained

3. The paper is currently ~4,500 words (excluding references), which is suitable for most journals

## Contact

For questions about the analysis or code, please contact [your email]