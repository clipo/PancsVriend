# TODO: Journal of Economic Interaction and Coordination Submission

## Key Submission Requirements (from JEIC-submission-guidelines.txt)

### Manuscript Format
- **LaTeX preferred** (use Springer Nature LaTeX template)
- **Double-blind peer review** - remove all author identifying information from manuscript
- **Abstract**: 150-250 words, no undefined abbreviations
- **Keywords**: 4-6 keywords for indexing
- **JEL codes**: Required (see https://www.aeaweb.org/econlit/jelCodes.php?view=jel)

### Citations and References
- **Citation style**: Author-year format (Thompson 1990)
- **Reference list**: Alphabetical by first author, include DOIs as full links
- **Journal abbreviations**: Use ISSN standard abbreviations

### Figures and Tables
- **Figures**: EPS for vector graphics, TIFF for halftones, minimum 300 dpi
- **Color**: Free online, contribution required for print
- **Captions**: Descriptive, begin with "Fig." in bold

### Required Statements
- **Competing Interests**: Must declare financial/non-financial interests
- **Data Availability Statement**: Required for all original research
- **Author Contributions**: Recommended for transparency

## High Priority Tasks

1. ✅ **Review submission guidelines** - COMPLETED

2. ✅ **Format manuscript using Springer template** - COMPLETED
   - Created new file: `schelling_llm_paper_springer.tex`
   - Uses `sn-jnl.cls` document class with `sn-basic` bibliography style
   - Follows double-blind peer review format (no author names in manuscript)
   - Includes proper JEL classification codes for economics journal
   - Abstract within 150-250 word limit
   - Contains required Declarations section

3. ✅ **Prepare manuscript content** - COMPLETED
   - ✅ Abstract written (189 words, within 150-250 limit)
   - ✅ References compiled in author-year format with DOIs (`references.bib`)

4. **Prepare submission materials**
   - Write cover letter explaining research contribution and fit to journal scope
   - Prepare supplementary materials (code, data, additional results)

5. **Submit manuscript**
   - Complete journal's online submission system and upload all materials

## Files Created

- `schelling_llm_paper_springer.tex` - Main manuscript file formatted for JEIC
- `references.bib` - Bibliography with 13 references in journal format
- Copy required template files: `sn-jnl.cls`, `sn-basic.bst` from `sn-article-template/`