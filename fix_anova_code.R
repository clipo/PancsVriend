# Fixed ANOVA code for the paper
# Run this if you're getting errors with the ANOVA analysis

library(tidyverse)
library(broom)

# Assuming you have metrics_long and metric_labels defined
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
         `p-value` = p_value, `Effect Size (η²)` = eta_squared)

# Alternative simpler approach if the above still gives errors:
# This calculates ANOVA for each metric separately
anova_results_simple <- data.frame()

for (m in unique(metrics_long$metric)) {
  data_subset <- metrics_long %>% filter(metric == m)
  aov_result <- aov(value ~ scenario, data = data_subset)
  aov_summary <- summary(aov_result)[[1]]
  
  # Extract F-statistic, p-value, and calculate eta squared
  f_stat <- aov_summary$`F value`[1]
  p_val <- aov_summary$`Pr(>F)`[1]
  df_between <- aov_summary$Df[1]
  df_within <- aov_summary$Df[2]
  ss_between <- aov_summary$`Sum Sq`[1]
  ss_total <- sum(aov_summary$`Sum Sq`)
  eta_sq <- ss_between / ss_total
  
  result_row <- data.frame(
    Metric = metric_labels[m],
    `F-statistic` = sprintf("%.2f", f_stat),
    `p-value` = ifelse(p_val < 0.001, "< 0.001", sprintf("%.3f", p_val)),
    `Effect Size (η²)` = round(eta_sq, 3),
    check.names = FALSE
  )
  
  anova_results_simple <- rbind(anova_results_simple, result_row)
}

# Use whichever works:
print("Method 1 results:")
print(anova_results)

print("\nMethod 2 results:")
print(anova_results_simple)