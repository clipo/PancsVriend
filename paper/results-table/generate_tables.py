#!/usr/bin/env python3
"""
Generate LaTeX tables from Schelling-Core-results-formatted.csv
Includes calculation of overall ordering metric
"""

import csv
import re
from collections import defaultdict

def parse_csv_data(filename):
    """Parse the CSV file and return structured data"""
    metrics_data = {}
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header row
        
        for row in reader:
            if not row[0]:  # Skip empty rows
                continue
            metric = row[0].strip()
            # Store the row data (excluding metric name)
            metrics_data[metric] = row[1:]
    
    return metrics_data

def parse_row_data(row_data):
    """Parse row data into structured format, grouping equal scenarios"""
    if not row_data:
        return []
    
    # Clean up the data - remove empty entries
    clean_data = [item.strip() for item in row_data if item.strip()]
    
    if not clean_data:
        return []
    
    # Parse alternating scenario,relation,scenario,relation... pattern
    result = []
    current_group = [clean_data[0]]  # Start with first scenario
    
    i = 1
    while i < len(clean_data):
        if i >= len(clean_data):
            break
            
        # Current item should be a relation
        relation = clean_data[i]
        i += 1
        
        if relation == '=':
            # Next item belongs to current group
            if i < len(clean_data):
                current_group.append(clean_data[i])
                i += 1
        else:
            # End current group, start new one with next scenario
            if i < len(clean_data):
                result.append((current_group, relation))
                current_group = [clean_data[i]]
                i += 1
            else:
                # No more scenarios, just append the relation to last group
                result.append((current_group, relation))
                current_group = []
    
    # Add final group if it exists
    if current_group:
        result.append((current_group, ''))
    
    return result

def get_scenario_order(metrics_data):
    """Extract the order of scenarios from the data"""
    # Use the first metric to get the scenario order
    first_metric = list(metrics_data.values())[0]
    parsed = parse_row_data(first_metric)
    
    scenarios = []
    for group, _ in parsed:
        scenarios.extend(group)
    
    return scenarios

def calculate_consensus_ordering(metrics_data, exclude_mechanical=False):
    """Calculate the overall consensus ordering"""
    all_scenarios = get_scenario_order(metrics_data)
    
    if exclude_mechanical:
        all_scenarios = [s for s in all_scenarios if s != 'mechanical']
    
    # Build adjacency pairs
    pairs = []
    for i in range(len(all_scenarios) - 1):
        pairs.append((all_scenarios[i], all_scenarios[i+1]))
    
    # For each pair, count agreements across metrics
    consensus = []
    
    for s1, s2 in pairs:
        less_count = 0
        greater_count = 0
        equal_count = 0
        
        for metric, row_data in metrics_data.items():
            parsed = parse_row_data(row_data)
            
            # Find positions of s1 and s2
            pos1, pos2 = None, None
            for i, (group, _) in enumerate(parsed):
                if s1 in group:
                    pos1 = i
                if s2 in group:
                    pos2 = i
            
            if pos1 is not None and pos2 is not None:
                if pos1 < pos2:
                    less_count += 1
                elif pos1 > pos2:
                    greater_count += 1
                else:  # Same group
                    equal_count += 1
        
        # Apply consensus rules
        total = len(metrics_data)
        if equal_count == total:
            consensus.append('=')
        elif less_count >= 5:
            consensus.append('<')
        elif less_count >= 1 and greater_count == 0:
            consensus.append('\\leq')
        elif greater_count > 0 and less_count > 0:
            consensus.append('\\leq^{?}')
        else:
            consensus.append('\\leq')  # Default to weak ordering
    
    # Build the overall ordering string
    result_parts = []
    scenario_groups = []
    current_group = [all_scenarios[0]]
    
    for i, relation in enumerate(consensus):
        if relation == '=':
            current_group.append(all_scenarios[i+1])
        else:
            scenario_groups.append((current_group, relation))
            current_group = [all_scenarios[i+1]]
    
    # Add last group
    scenario_groups.append((current_group, ''))
    
    return scenario_groups

def format_scenario_group(scenarios, output_format='latex'):
    """Format a group of scenarios for LaTeX or plain text"""
    if output_format == 'latex':
        if len(scenarios) == 1:
            escaped = scenarios[0].replace('_', '\\_')
            return f"\\begin{{pmatrix}} \\text{{{escaped}}} \\end{{pmatrix}}"
        else:
            formatted = []
            for s in sorted(scenarios):
                escaped = s.replace('_', '\\_')
                formatted.append(f"\\text{{{escaped}}}")
            separator = ' \\\\ '
            return f"\\begin{{pmatrix}} {separator.join(formatted)} \\end{{pmatrix}}"
    else:  # plain text
        if len(scenarios) == 1:
            return scenarios[0]
        else:
            return "(" + ", ".join(sorted(scenarios)) + ")"

def generate_latex_row(metric, row_data, num_terms, max_terms):
    """Generate a LaTeX row for a metric"""
    parsed = parse_row_data(row_data)
    
    # Calculate centering offset
    offset = (max_terms - num_terms) // 2
    
    # Build row
    metric_escaped = metric.replace('_', '\\_')
    row_parts = [f"\\text{{{metric_escaped}:}}"]
    
    # Add empty cells for centering
    for _ in range(offset):
        row_parts.extend(['', ''])
    
    # Add the actual content
    for i, (group, relation) in enumerate(parsed):
        if i > 0:
            # Format relation with stars as superscript
            rel_formatted = relation.replace('*', '')
            stars = '*' * relation.count('*')
            if stars:
                row_parts.append(f"{rel_formatted}^{{{stars}}}")
            else:
                row_parts.append(rel_formatted)
        
        row_parts.append(format_scenario_group(group))
    
    # Pad if necessary to maintain alignment
    while len(row_parts) < 2 * max_terms:
        row_parts.append('')
    
    return ' & '.join(row_parts)

def filter_mechanical_from_parsed(parsed_data):
    """Filter out mechanical scenario from already parsed data"""
    filtered = []
    
    for i, (group, relation) in enumerate(parsed_data):
        # Remove mechanical from the group
        filtered_group = [s for s in group if s != 'mechanical']
        
        if filtered_group:  # Only add if group is not empty
            if i == len(parsed_data) - 1:
                # Last group - no relation
                filtered.append((filtered_group, ''))
            else:
                # Check if next group exists and adjust relation accordingly
                next_group_has_non_mechanical = False
                for j in range(i + 1, len(parsed_data)):
                    next_filtered_group = [s for s in parsed_data[j][0] if s != 'mechanical']
                    if next_filtered_group:
                        next_group_has_non_mechanical = True
                        break
                
                if next_group_has_non_mechanical:
                    filtered.append((filtered_group, relation))
                else:
                    filtered.append((filtered_group, ''))
    
    return filtered

def generate_latex_table(metrics_data, overall_ordering, exclude_mechanical=False):
    
    # Build table
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    
    if exclude_mechanical:
        lines.append("\\caption{Segregation Metrics Comparison Across LLM Scenarios}")
        lines.append("\\label{tab:segregation_metrics_llm}")
    else:
        lines.append("\\caption{Segregation Metrics Comparison Across Scenarios}")
        lines.append("\\label{tab:segregation_metrics}")
    
    lines.append("\\vspace{1.5em}")
    lines.append("\\begin{align*}")
    
    # Add overall ordering as first row
    overall_line = generate_overall_ordering_row(overall_ordering)
    lines.append(overall_line + " \\\\\\\\[2.5em]")
    
    # Add horizontal line after overall
    lines.append("\\hline \\\\\\\\[0.5em]")
    
    # Add metrics
    metric_order = ['-clusters', 'distance', 'ghetto_rate', 'mix_deviation', 'share', '-switch_rate']
    
    for i, metric in enumerate(metric_order):
        if metric in metrics_data:
            parsed = parse_row_data(metrics_data[metric])
            
            # Filter mechanical if needed
            if exclude_mechanical:
                parsed = filter_mechanical_from_parsed(parsed)
            
            row = generate_align_row(metric, parsed)
            
            # Determine spacing based on content
            if any(len(group) >= 3 for group, _ in parsed):
                spacing = "2em"
            else:
                spacing = "1em"
            
            if i < len(metric_order) - 1:
                lines.append(row + f" \\\\\\\\[{spacing}]")
            else:
                lines.append(row)
    
    lines.append("\\end{align*}")
    lines.append("\\vspace{2em}")
    lines.append("")
    lines.append("\\small{Six metrics where more of the metric is associated with higher segregation.\\\\")
    lines.append("Higher clusters and switch\\_rate are associated with \\textit{less} segregation, so -clusters and -switch\\_rate are presented here for easier comparison.\\\\")
    lines.append("Note: There is general agreement with the order.}")
    lines.append("\\end{table}")
    
    return '\n'.join(lines)

def generate_overall_ordering_row(overall_ordering):
    """Generate the overall ordering row in align* format"""
    parts = ["\\text{overall:} & \\quad "]
    
    for i, (group, relation) in enumerate(overall_ordering):
        # Add the group
        parts.append(format_scenario_group(group, 'latex'))
        
        # Add relation after this group if it exists
        if relation:
            parts.append(f" & {relation} & ")
        elif i < len(overall_ordering) - 1:
            # No relation but not the last group - add empty relation
            parts.append(" &  & ")
    
    return "".join(parts)

def generate_align_row(metric, parsed_data):
    """Generate a metric row in align* format"""
    metric_escaped = metric.replace('_', '\\_')
    parts = [f"\\text{{{metric_escaped}:}} & \\quad "]
    
    for i, (group, relation) in enumerate(parsed_data):
        # Add the group
        parts.append(format_scenario_group(group, 'latex'))
        
        # Add relation after this group if it exists
        if relation:
            # Format relation with stars as superscript
            if relation.startswith('<') and '*' in relation:
                rel_base = relation.replace('*', '')
                stars = '*' * relation.count('*')
                formatted_rel = f"{rel_base}^{{{stars}}}"
            else:
                formatted_rel = relation
            parts.append(f" & {formatted_rel} & ")
        elif i < len(parsed_data) - 1:
            # No relation but not the last group - add empty relation
            parts.append(" &  & ")
    
    return "".join(parts)

def generate_plain_text_table(metrics_data, overall_ordering, exclude_mechanical=False):
    """Generate plain text version of the table"""
    lines = []
    
    # Title
    if exclude_mechanical:
        lines.append("SEGREGATION METRICS COMPARISON ACROSS LLM SCENARIOS")
    else:
        lines.append("SEGREGATION METRICS COMPARISON ACROSS SCENARIOS")
    lines.append("=" * len(lines[-1]))
    lines.append("")
    
    # Overall ordering
    overall_line = generate_plain_text_ordering_row(overall_ordering)
    lines.append("OVERALL CONSENSUS ORDERING:")
    lines.append(overall_line)
    lines.append("")
    lines.append("-" * 80)
    lines.append("")
    
    # Individual metrics
    metric_order = ['-clusters', 'distance', 'ghetto_rate', 'mix_deviation', 'share', '-switch_rate']
    
    for metric in metric_order:
        if metric in metrics_data:
            parsed = parse_row_data(metrics_data[metric])
            
            # Filter mechanical if needed
            if exclude_mechanical:
                parsed = filter_mechanical_from_parsed(parsed)
            
            row = generate_plain_text_metric_row(metric, parsed)
            lines.append(row)
            lines.append("")
    
    # Footer explanation
    lines.append("-" * 80)
    lines.append("NOTES:")
    lines.append("- Six metrics where more of the metric is associated with higher segregation")
    lines.append("- Higher clusters and switch_rate are associated with LESS segregation,")
    lines.append("  so -clusters and -switch_rate are presented for easier comparison")
    lines.append("- Significance levels: * p<0.05, ** p<0.01, *** p<0.001")
    lines.append("- Inequality symbols: < (strong consensus), ≤ (weak consensus), ≤? (mixed evidence)")
    
    return '\n'.join(lines)

def generate_plain_text_ordering_row(overall_ordering):
    """Generate plain text version of overall ordering"""
    parts = []
    
    for i, (group, relation) in enumerate(overall_ordering):
        group_text = format_scenario_group(group, 'text')
        parts.append(group_text)
        
        if relation:
            # Convert LaTeX symbols to text equivalents
            relation_text = relation.replace('\\leq', '≤').replace('\\leq^{?}', '≤?')
            parts.append(f" {relation_text} ")
    
    return "".join(parts)

def generate_plain_text_metric_row(metric, parsed_data):
    """Generate plain text version of a metric row"""
    metric_clean = metric.replace('_', '_').replace('-', '-')
    parts = [f"{metric_clean:15}: "]
    
    for i, (group, relation) in enumerate(parsed_data):
        # Add the group
        group_text = format_scenario_group(group, 'text')
        parts.append(group_text)
        
        # Add relation after this group if it exists
        if relation:
            # Format relation with stars
            if relation.startswith('<') and '*' in relation:
                rel_base = relation.replace('*', '')
                stars = '*' * relation.count('*')
                formatted_rel = f" {rel_base}^{stars} "
            else:
                formatted_rel = f" {relation} "
            parts.append(formatted_rel)
        elif i < len(parsed_data) - 1:
            # No relation but not the last group - add space
            parts.append(" ")
    
    return "".join(parts)

def main():
    """Main function"""
    # Parse CSV data
    csv_file = 'Schelling-Core-results-formatted.csv'
    metrics_data = parse_csv_data(csv_file)
    
    # Calculate overall orderings
    overall_all = calculate_consensus_ordering(metrics_data, exclude_mechanical=False)
    overall_llm = calculate_consensus_ordering(metrics_data, exclude_mechanical=True)
    
    # Generate LaTeX tables
    table_all = generate_latex_table(metrics_data, overall_all, exclude_mechanical=False)
    table_llm = generate_latex_table(metrics_data, overall_llm, exclude_mechanical=True)
    
    # Generate plain text tables
    text_all = generate_plain_text_table(metrics_data, overall_all, exclude_mechanical=False)
    text_llm = generate_plain_text_table(metrics_data, overall_llm, exclude_mechanical=True)
    
    # Write LaTeX files
    with open('results_table.tex', 'w') as f:
        f.write(table_all)
    
    with open('results_table_llmonly.tex', 'w') as f:
        f.write(table_llm)
    
    # Write plain text files
    with open('results_table.txt', 'w') as f:
        f.write(text_all)
    
    with open('results_table_llmonly.txt', 'w') as f:
        f.write(text_llm)
    
    print("Generated LaTeX files: results_table.tex and results_table_llmonly.tex")
    print("Generated text files: results_table.txt and results_table_llmonly.txt")
    
    # Print overall orderings for verification
    print("\nOverall ordering (all scenarios):")
    for group, rel in overall_all:
        rel_display = rel.replace('\\geq', '≥').replace('\\leq^{?}', '≤?')
        if rel:
            print(f"  {group} {rel_display}", end=" ")
        else:
            print(f"  {group}")
    
    print("\nOverall ordering (LLM only):")
    for group, rel in overall_llm:
        rel_display = rel.replace('\\geq', '≥').replace('\\leq^{?}', '≤?')
        if rel:
            print(f"  {group} {rel_display}", end=" ")
        else:
            print(f"  {group}")

if __name__ == "__main__":
    main()