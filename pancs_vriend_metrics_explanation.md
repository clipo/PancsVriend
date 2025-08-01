# The Pancs-Vriend Segregation Metrics: A Comprehensive Guide

## Background

Pancs and Vriend (2007) developed a comprehensive framework of six complementary metrics specifically designed for measuring segregation in grid-based agent models. Unlike traditional urban segregation indices (like the dissimilarity index), these metrics are tailored for the small-scale, discrete nature of agent-based models where traditional continuous measures may be inappropriate.

## The Six Metrics Explained

### 1. **Share** (Global Segregation Level)
**What it measures**: The proportion of all neighbor pairs that are of the same type.

**Calculation**: 
- Count all adjacent agent pairs (horizontally, vertically, and diagonally)
- Calculate: (same-type pairs) / (total pairs)

**Interpretation**:
- 0.5 = Perfect integration (random mixing)
- 1.0 = Complete segregation (no cross-type neighbors)
- Values > 0.5 indicate segregation

**Example**: In a perfectly integrated checkerboard pattern, share = 0.0. In completely segregated halves, share = 1.0.

### 2. **Clusters** (Spatial Fragmentation)
**What it measures**: The number of spatially contiguous regions of same-type agents.

**Calculation**:
- Use depth-first search to identify connected components
- Count distinct regions where same-type agents are adjacent (4-connectivity)

**Interpretation**:
- Lower values = Larger, consolidated ethnic enclaves
- Higher values = More fragmented, scattered communities
- 1 cluster per type = Maximum segregation
- Many small clusters = More integrated but scattered

**Example**: Two completely separated halves = 2 clusters. A salt-and-pepper pattern might have 20+ clusters.

### 3. **Distance** (Spatial Separation)
**What it measures**: Average Manhattan distance from each agent to their nearest different-type neighbor.

**Calculation**:
- For each agent, find the nearest agent of a different type
- Calculate Manhattan distance (|x1-x2| + |y1-y2|)
- Average across all agents

**Interpretation**:
- Higher values = Groups are spatially separated
- Lower values = Groups are intermixed
- Depends on grid size (normalize for comparison)

**Example**: Adjacent different-type neighbors = distance 1. Opposite corners of a 15×15 grid = distance 28.

### 4. **Mix Deviation** (Local Integration Quality)
**What it measures**: How much each agent's neighborhood deviates from perfect 50-50 integration.

**Calculation**:
- For each agent, count like vs unlike neighbors (8-neighborhood)
- Calculate: |0.5 - (like neighbors / total neighbors)|
- Average across all agents

**Interpretation**:
- 0.0 = Perfect local integration (all agents have 50% like neighbors)
- 0.5 = Complete segregation (all agents have 100% like neighbors)
- Measures segregation at the individual level

**Example**: An agent with 6 like and 2 unlike neighbors has deviation = |0.5 - 0.75| = 0.25.

### 5. **Switch Rate** (Border Roughness)
**What it measures**: The frequency of type changes along borders between agents.

**Calculation**:
- For each agent with neighbors, trace around its neighbors
- Count type switches in the sequence
- Calculate: (switches) / (total border segments)

**Interpretation**:
- Higher values = More jagged, intermixed borders
- Lower values = Smooth, consolidated boundaries
- 0.0 = Perfect segregation with smooth borders
- High values indicate "salt-and-pepper" patterns

**Example**: A straight border between two groups has switch rate ≈ 0. A checkerboard has maximum switch rate.

### 6. **Ghetto Rate** (Extreme Segregation)
**What it measures**: The count of agents living in completely homogeneous neighborhoods.

**Calculation**:
- Count agents with zero different-type neighbors (8-neighborhood)
- No normalization (raw count)

**Interpretation**:
- Captures extreme segregation pockets
- High values indicate "ghettoization"
- More sensitive than share to complete isolation
- Critical for policy implications

**Example**: An agent surrounded by 8 same-type neighbors contributes 1 to ghetto rate.

## Why Six Metrics?

Each metric captures different aspects of segregation:

1. **Share** gives the overall level but misses spatial patterns
2. **Clusters** shows fragmentation but not density
3. **Distance** measures separation but not local mixing
4. **Mix Deviation** focuses on individual experiences
5. **Switch Rate** captures boundary complexity
6. **Ghetto Rate** identifies extreme isolation

Together, they provide a comprehensive picture that no single metric could capture.

## Advantages Over Traditional Indices

1. **Designed for grids**: Unlike continuous urban indices, these work with discrete agent locations
2. **Multiple dimensions**: Capture both global patterns and local experiences
3. **Computationally efficient**: All can be calculated in one grid pass
4. **Intuitive interpretation**: Each has clear meaning for policy makers
5. **Sensitive to different patterns**: Can distinguish between different types of segregation

## In Our Study

We use all six metrics to reveal how different social contexts produce qualitatively different segregation patterns:

- **Political contexts**: High ghetto rate (61.6) with low switch rate (0.076) = consolidated, extreme segregation
- **Economic contexts**: Low ghetto rate (5.0) with high switch rate (0.471) = fragmented, fluid patterns
- **Racial contexts**: Intermediate values that match real-world urban segregation indices

The metrics together show not just "how much" segregation, but "what kind" - critical for understanding the bias paradox in LLMs.