# R Code for "Social Context Matters: How Large Language Model Agents 
# Reproduce Real-World Segregation Patterns in the Schelling Model"
# 
# This file contains all R code used for data analysis and visualization
# in the paper. Code sections correspond to specific parts of the manuscript.
#
# Paper file: schelling_llm_paper_updated_JEIC.tex
# Author: [To be added after acceptance - removed for double-blind review]

# =============================================================================
# SECTION 1: Data Preparation and Visualization Setup
# Corresponds to: Section 3.1 "Overall Segregation Patterns" 
# Figure 1: Comprehensive segregation patterns
# =============================================================================

# Prepare data for visualization
metrics_long <- combined_data %>%
  select(scenario, scenario_label, clusters, switch_rate, distance, 
         mix_deviation, share, ghetto_rate) %>%
  pivot_longer(cols = c(clusters, switch_rate, distance, mix_deviation, share, ghetto_rate),
               names_to = "metric", values_to = "value")

# Define metric labels
metric_labels <- c(
  'clusters' = 'Number of Clusters',
  'switch_rate' = 'Switch Rate',
  'distance' = 'Average Distance',
  'mix_deviation' = 'Mix Deviation',
  'share' = 'Segregation Share',
  'ghetto_rate' = 'Ghetto Formation Rate'
)

# Create faceted plot
metrics_plot <- metrics_long %>%
  group_by(scenario, scenario_label, metric) %>%
  summarise(
    mean = mean(value),
    sd = sd(value),
    se = sd(value) / sqrt(n()),
    .groups = "drop"
  ) %>%
  mutate(metric_label = metric_labels[metric]) %>%
  ggplot(aes(x = scenario_label, y = mean, fill = scenario_label)) +
  geom_col(alpha = 0.8) +
  geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.2) +
  facet_wrap(~ metric_label, scales = "free_y", ncol = 2) +
  scale_fill_manual(values = scenario_colors) +
  labs(x = "", y = "Metric Value", 
       title = "Segregation Patterns Across Social Contexts") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")

metrics_plot

# =============================================================================
# SECTION 2: Summary Statistics Table
# Corresponds to: Section 3.2 "Key Findings by Context"
# Table 1: Summary statistics for key segregation metrics
# =============================================================================

summary_table <- combined_data %>%
  group_by(scenario_label) %>%
  summarise(
    `Ghetto Rate` = sprintf("%.1f ± %.1f", mean(ghetto_rate), sd(ghetto_rate)),
    `Seg. Share` = sprintf("%.3f ± %.3f", mean(share), sd(share)),
    `Distance` = sprintf("%.2f ± %.2f", mean(distance), sd(distance)),
    `Switch Rate` = sprintf("%.3f ± %.3f", mean(switch_rate), sd(switch_rate)),
    N = n()
  ) %>%
  rename(Context = scenario_label)

# =============================================================================
# SECTION 3: Statistical Analysis (ANOVA)
# Corresponds to: Section 3.3 "Statistical Significance"
# Table 2: ANOVA results comparing segregation metrics
# =============================================================================

# Perform ANOVA for each metric
anova_results <- metrics_long %>%
  group_by(metric) %>%
  do({
    aov_result <- aov(value ~ scenario, data = .)
    tidy_result <- tidy(aov_result)
    scenario_row <- tidy_result[tidy_result$term == "scenario", ]
    residual_row <- tidy_result[tidy_result$term == "Residuals", ]
    
    data.frame(
      metric = unique(.$metric),
      statistic = scenario_row$statistic,
      p.value = scenario_row$p.value,
      df_scenario = scenario_row$df,
      df_residual = residual_row$df
    )
  }) %>%
  ungroup() %>%
  mutate(
    metric_label = metric_labels[metric],
    F_statistic = sprintf("%.2f", statistic),
    p_value = ifelse(p.value < 0.001, "< 0.001", sprintf("%.3f", p.value)),
    eta_squared = round((df_scenario * statistic) / (df_scenario * statistic + df_residual), 3)
  ) %>%
  select(Metric = metric_label, `F-statistic` = F_statistic, 
         `p-value` = p_value, `Effect Size (eta^2)` = eta_squared)

# =============================================================================
# SECTION 4: Convergence Analysis
# Corresponds to: Section 3.4 "Convergence Patterns"
# Figure 2: Convergence dynamics across social contexts
# =============================================================================

# Create convergence data (based on the analysis results)
convergence_data <- data.frame(
  scenario = c('baseline', 'ethnic_asian_hispanic', 'income_high_low', 
               'political_liberal_conservative', 'race_white_black'),
  mean_steps = c(73.2, 77.4, 6.5, 62.0, 81.8),
  sd_steps = c(57.7, 63.7, 4.1, 43.3, 55.9)
) %>%
  mutate(scenario_label = scenario_labels[scenario])

conv_plot <- ggplot(convergence_data, aes(x = reorder(scenario_label, mean_steps), 
                                          y = mean_steps, fill = scenario_label)) +
  geom_col(alpha = 0.8) +
  geom_errorbar(aes(ymin = mean_steps - sd_steps/sqrt(10), 
                    ymax = mean_steps + sd_steps/sqrt(10)), width = 0.2) +
  scale_fill_manual(values = scenario_colors) +
  labs(x = "Social Context", y = "Steps to 90% Convergence",
       title = "Convergence Speed by Social Context") +
  theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1))

conv_plot

# =============================================================================
# SECTION 5: Segregation Heatmap Visualization
# Corresponds to: Section 3.5 "Visualization: Segregation Heatmap"
# Figure 3: Normalized segregation intensity heatmap
# =============================================================================

# Create normalized heatmap data
heatmap_data <- combined_data %>%
  group_by(scenario) %>%
  summarise(across(c(clusters, switch_rate, distance, mix_deviation, share, ghetto_rate), mean)) %>%
  pivot_longer(cols = -scenario, names_to = "metric", values_to = "value") %>%
  group_by(metric) %>%
  mutate(normalized = (value - min(value)) / (max(value) - min(value))) %>%
  ungroup() %>%
  mutate(
    scenario_label = scenario_labels[scenario],
    metric_label = metric_labels[metric]
  )

# Order scenarios by overall segregation level
scenario_order <- c("Income\n(High/Low)", "Baseline\n(Red/Blue)", 
                   "Race\n(White/Black)", "Ethnic\n(Asian/Hispanic)", 
                   "Political\n(Liberal/Conservative)")

heatmap_plot <- ggplot(heatmap_data, 
                      aes(x = metric_label, y = factor(scenario_label, levels = scenario_order), 
                          fill = normalized)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.2f", normalized)), 
            color = ifelse(heatmap_data$normalized > 0.5, "white", "black")) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0.5,
                      name = "Normalized\nSegregation") +
  labs(x = "", y = "", title = "Segregation Intensity Across Contexts and Metrics") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

heatmap_plot

# =============================================================================
# SECTION 6: Appendix - Baseline Comparison Analysis
# Corresponds to: Appendix "Detailed Statistical Results"
# Table 3: Pairwise comparisons between baseline and other social contexts
# =============================================================================

# Create comparison table
comparison_stats <- combined_data %>%
  filter(scenario == "baseline") %>%
  select(baseline_ghetto = ghetto_rate, baseline_share = share) %>%
  summarise(
    baseline_ghetto_mean = mean(baseline_ghetto),
    baseline_share_mean = mean(baseline_share)
  ) %>%
  cross_join(
    combined_data %>%
      filter(scenario != "baseline") %>%
      group_by(scenario_label) %>%
      summarise(
        ghetto_mean = mean(ghetto_rate),
        ghetto_sd = sd(ghetto_rate),
        share_mean = mean(share),
        share_sd = sd(share),
        n = n()
      )
  ) %>%
  mutate(
    ghetto_diff_pct = (ghetto_mean - baseline_ghetto_mean) / baseline_ghetto_mean * 100,
    share_diff_pct = (share_mean - baseline_share_mean) / baseline_share_mean * 100
  ) %>%
  select(
    Context = scenario_label,
    `Ghetto Rate` = ghetto_mean,
    `Ghetto vs Baseline` = ghetto_diff_pct,
    `Share` = share_mean,
    `Share vs Baseline` = share_diff_pct
  ) %>%
  mutate(
    `Ghetto Rate` = sprintf("%.1f", `Ghetto Rate`),
    `Ghetto vs Baseline` = sprintf("%+.0f%%", `Ghetto vs Baseline`),
    `Share` = sprintf("%.3f", `Share`),
    `Share vs Baseline` = sprintf("%+.1f%%", `Share vs Baseline`)
  )

# =============================================================================
# ADDITIONAL ANALYSIS FUNCTIONS AND UTILITIES
# =============================================================================

# Function to create scenario color palette (define if not already available)
# scenario_colors <- c(
#   "Baseline\n(Red/Blue)" = "#1f77b4",
#   "Race\n(White/Black)" = "#ff7f0e", 
#   "Ethnic\n(Asian/Hispanic)" = "#2ca02c",
#   "Income\n(High/Low)" = "#d62728",
#   "Political\n(Liberal/Conservative)" = "#9467bd"
# )

# Function to create scenario labels mapping (define if not already available)
# scenario_labels <- c(
#   "baseline" = "Baseline\n(Red/Blue)",
#   "race_white_black" = "Race\n(White/Black)",
#   "ethnic_asian_hispanic" = "Ethnic\n(Asian/Hispanic)", 
#   "income_high_low" = "Income\n(High/Low)",
#   "political_liberal_conservative" = "Political\n(Liberal/Conservative)"
# )

# =============================================================================
# SESSION INFO AND PACKAGE REQUIREMENTS
# =============================================================================

# Required packages for analysis:
# library(tidyverse)   # Data manipulation and visualization
# library(ggplot2)     # Plotting
# library(dplyr)       # Data manipulation
# library(tidyr)       # Data reshaping
# library(broom)       # Statistical output tidying
# library(knitr)       # Table formatting (if needed for external output)

# To reproduce this analysis:
# 1. Load the required packages listed above
# 2. Ensure 'combined_data' contains the simulation results with columns:
#    - scenario, scenario_label, clusters, switch_rate, distance
#    - mix_deviation, share, ghetto_rate
# 3. Define scenario_colors and scenario_labels mappings
# 4. Run the code sections in order

# End of R analysis code