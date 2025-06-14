#!/usr/bin/env Rscript
# Verify that the paper data files exist and can be loaded

cat("Checking paper data files...\n\n")

# Check if files exist
files_to_check <- c(
  "convergence_analysis_detailed.csv",
  "pairwise_comparison_results.csv",
  "schelling_llm_paper.qmd",
  "references.bib"
)

all_exist <- TRUE
for (file in files_to_check) {
  if (file.exists(file)) {
    cat(sprintf("✓ %s exists\n", file))
  } else {
    cat(sprintf("✗ %s NOT FOUND\n", file))
    all_exist <- FALSE
  }
}

cat("\n")

# Try loading the data
if (all_exist) {
  cat("Loading data files...\n")
  
  tryCatch({
    convergence_data <- read.csv("convergence_analysis_detailed.csv")
    cat(sprintf("✓ Convergence data loaded: %d rows, %d columns\n", 
                nrow(convergence_data), ncol(convergence_data)))
    
    pairwise_data <- read.csv("pairwise_comparison_results.csv")
    cat(sprintf("✓ Pairwise data loaded: %d rows, %d columns\n", 
                nrow(pairwise_data), ncol(pairwise_data)))
    
    cat("\nData summary:\n")
    cat("Convergence experiments:", unique(convergence_data$experiment), "\n")
    cat("Metrics analyzed:", unique(pairwise_data$metric), "\n")
    
  }, error = function(e) {
    cat("✗ Error loading data:", e$message, "\n")
  })
  
} else {
  cat("✗ Cannot load data - some files are missing\n")
}

cat("\nTo render the paper, run:\n")
cat("  quarto render schelling_llm_paper.qmd\n")