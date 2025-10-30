# Segregation Metrics Analysis: Automated Table Generation

This directory contains a complete automated pipeline for generating publication-ready segregation metrics tables from CSV data.

## Overview

The system converts CSV data containing segregation metric comparisons into formatted LaTeX tables with:
- Overall consensus ordering derived from cross-metric agreement
- Individual metric comparisons with statistical significance
- Comprehensive legend and documentation
- Publication-ready PDF output

## Files

### Core Data and Generated Tables
- `Schelling-Core-results-formatted.csv` - Source data with segregation metrics
- `results_table.tex` - Complete analysis (all scenarios) - LaTeX format
- `results_table_llmonly.tex` - LLM-only analysis (excludes mechanical) - LaTeX format
- `results_table.txt` - Complete analysis (all scenarios) - Plain text format
- `results_table_llmonly.txt` - LLM-only analysis (excludes mechanical) - Plain text format

### Documentation and Legend  
- `legend_glossary.tex` - Comprehensive legend explaining scenarios, metrics, and symbols - LaTeX format
- `legend_glossary.txt` - Comprehensive legend explaining scenarios, metrics, and symbols - Plain text format
- `show_legend.tex` - Standalone legend document
- `show_legend.pdf` - Compiled legend

### Comprehensive Output
- `comprehensive_results.tex` - Master document combining tables + legend
- `comprehensive_results.pdf` - Complete analysis with full documentation (8 pages)

### Automation
- `generate_tables.py` - Main script for automated table generation
- `show_results_table.tex` - Simple wrapper for viewing basic tables

## Quick Start

### Generate Tables from CSV
```bash
python generate_tables.py
```
This generates both LaTeX (.tex) and plain text (.txt) versions of the tables.

### Compile Comprehensive Document  
```bash
pdflatex comprehensive_results.tex
```

### View Individual Components
```bash
# Basic tables only (LaTeX)
pdflatex show_results_table.tex

# Basic tables (plain text)
cat results_table.txt
cat results_table_llmonly.txt

# Legend only (LaTeX)
pdflatex show_legend.tex

# Legend only (plain text)
cat legend_glossary.txt
```

## Data Format

The CSV file uses alternating scenario-relation-scenario-relation pattern:
```
Variable,,,,,,,,,,,,,
-clusters,income,=,green/yellow,<***,llm_baseline,<***,race:wht/blk,<***,ethnicity,<***,mechanical,=,political
```

## Social Context Scenarios

| Scenario | Description |
|----------|-------------|
| `green/yellow` | Baseline control (red vs blue teams) |
| `race:wht/blk` | White middle-class vs Black families |
| `ethnicity` | Asian American vs Hispanic/Latino families |
| `income` | High-income professionals vs working-class families |
| `political` | Liberal vs conservative households |
| `llm_baseline` | LLM agents using baseline scenario |
| `mechanical` | Traditional utility-maximizing agents |

## Segregation Metrics (Pancs-Vriend Framework)

| Metric | Description |
|--------|-------------|
| `Share` | Proportion of same-type neighbor pairs (global segregation) |
| `Clusters` | Number of contiguous same-type regions (presented as -clusters) |
| `Distance` | Average distance to nearest different-type agent |
| `Ghetto Rate` | Count of agents with zero different-type neighbors |
| `Mix Deviation` | Deviation from 50-50 local integration |
| `Switch Rate` | Frequency of type changes along borders (presented as -switch_rate) |

## Consensus Ordering Algorithm

The overall ordering synthesizes evidence across all six metrics:

1. **Count agreements** between adjacent scenario pairs across metrics
2. **Apply consensus rules**:
   - `<`: Strong consensus (≥5 metrics agree)
   - `≤`: Weak consensus (≥1 metric agrees, no disagreement)  
   - `≤?`: Mixed evidence (disagreements exist)
   - `=`: Perfect agreement (all metrics agree)
3. **Group equal scenarios** vertically using pmatrix formatting

## Key Research Findings

- **Political contexts** show extreme segregation (ghetto rate: 61.6, share: 0.928)
- **Economic contexts** show minimal clustering (ghetto rate: 5.0, share: 0.543)  
- **12.3× difference** in ghetto formation based purely on social framing
- All scenarios differ significantly from baseline (p < 0.001)

## Customization

### Modifying Consensus Rules
Edit the `calculate_consensus_ordering()` function in `generate_tables.py`:
```python
# Current rules
elif less_count >= 5:
    consensus.append('<')
elif less_count >= 1 and greater_count == 0:
    consensus.append('\\leq')
```

### Adding New Scenarios
1. Add data to CSV following alternating pattern
2. Update scenario descriptions in `legend_glossary.tex`
3. Regenerate tables with `python generate_tables.py`

### Output Formatting
- Modify LaTeX formatting in `generate_latex_table()` function
- Adjust spacing, fonts, and layout in the template functions
- Update `comprehensive_results.tex` for document structure

## Dependencies

- Python 3.x with csv module
- LaTeX distribution with:
  - amsmath, amssymb, amsfonts
  - booktabs, geometry
  - float, afterpage

## Troubleshooting

### Common Issues
- **Unicode errors**: Use LaTeX math mode (`$\leq$`) instead of Unicode (≤)
- **Float too large**: Tables may exceed page width - adjust geometry or content
- **Missing relations**: Check CSV parsing if final relations are dropped

### Validation
Run the complete pipeline to verify all components work:
```bash
python generate_tables.py && pdflatex comprehensive_results.tex
```

## Citation

For complete methodological details, see:
*Social Context Matters: How Large Language Model Agents Reproduce Real-World Segregation Patterns in the Schelling Model* (Pancs & Vriend, submitted to Journal of Economic Interaction and Coordination).