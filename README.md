# BASELINE
## Builder Base Strategy & Linear Integer Engine

A prescriptive optimization system for strategic attack planning in Clash of Clans Builder Base scenarios. BASELINE employs deterministic computer vision techniques and mixed-integer linear programming to generate optimal attack strategies from base layout screenshots.

---

## Abstract

BASELINE addresses the resource allocation problem of selecting an optimal troop composition that maximizes weighted defense neutralization, subject to housing capacity and combat effectiveness constraints. The system transforms raw visual input (screenshots) into a structured optimization problem through a deterministic computer vision pipeline, then solves for the optimal troop mix using mixed-integer linear programming.

**Key Contributions:**
- Deterministic vision pipeline using OpenCV template matching (no deep learning dependencies)
- Graph-based spatial representation of defense relationships
- Heuristic threat scoring with clustering-based weight adjustment
- MILP formulation for prescriptive troop composition selection
- Effectiveness-based combat modeling with linear constraints
- Interactive manual editing interface for human-in-the-loop refinement

---

## Problem Formulation: Prescriptive Optimization

### Problem Statement

Given a Builder Base layout with n defense structures and m available troop types, determine the optimal troop composition such that:

1. **Objective**: Maximize total weighted defense neutralization
2. **Constraints**: 
   - Housing capacity limit (total troop housing space)
   - Damage sufficiency (troops must deal sufficient damage to neutralize selected defenses)
   - Survivability (optional: total troop health must withstand defense damage over estimated combat duration)
3. **Input**: Screenshot image of the base layout
4. **Output**: Optimal troop composition (counts per type) and predicted neutralized defenses

This is fundamentally a **resource allocation problem** formulated as a mixed-integer linear program, where:
- Resources = troop types with associated costs (housing) and capabilities (DPS, health)
- Targets = defense structures with health, DPS, and strategic weights
- Allocation = integer troop counts and binary defense neutralization indicators
- Objective = maximize weighted sum of neutralized defenses

### Prescriptive vs. Predictive Approach

Unlike predictive models that forecast outcomes, BASELINE is **prescriptive**: it recommends specific actions (which troops to deploy) that optimize a defined objective function subject to operational constraints. The system combines:

- **Descriptive analytics**: "What defenses exist and where?" (computer vision)
- **Prescriptive analytics**: "Which troop composition maximizes neutralized threat?" (optimization)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                              │
│                    Screenshot Image (RGB)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   COMPUTER VISION PIPELINE                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Preprocessing│→ │   Template   │→ │Postprocessing│          │
│  │  (OpenCV)    │  │   Matching   │  │ (Clustering) │          │
│  │              │  │  (TM_CCOEFF) │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                  │                  │                 │
│         └──────────────────┴──────────────────┘                 │
│                            │                                     │
│                    Defense Detections                            │
│              (type, x, y, confidence)                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA REPRESENTATION                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Defense    │→ │  Spatial      │→ │  Threat      │          │
│  │   Objects    │  │   Graph       │  │  Scoring     │          │
│  │  (Schema)    │  │  (NetworkX)   │  │  (Heuristic) │          │
│  │ health, DPS  │  │               │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐                                              │
│  │   Troop      │                                              │
│  │   Types      │                                              │
│  │  (Stats)     │                                              │
│  └──────────────┘                                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   OPTIMIZATION ENGINE                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Mixed-Integer Linear Program (MILP)                │  │
│  │                                                            │  │
│  │  Decision Variables:                                       │  │
│  │    x[i] ∈ Z≥0  (troop counts)                              │  │
│  │    y[j] ∈ {0,1} (defense neutralization)                    │  │
│  │  Objective: maximize Σ(w[j] × y[j])                        │  │
│  │  Constraints:                                              │  │
│  │    Housing: Σ(c[i] × x[i]) ≤ B                            │  │
│  │    Damage: Σ(d[i] × x[i] × e[i,j]) ≥ H[j] × y[j]          │  │
│  │    Survivability: Σ(h[i] × x[i]) ≥ Σ(D[j] × y[j] × T)     │  │
│  │  Solver: PuLP (CBC backend)                                │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                               │
│         Troop Composition (Counts per Type)                      │
│         Predicted Neutralized Defenses                            │
│         Visualization & Interactive UI                           │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interactions

1. **Vision → Data**: Detections converted to typed `Defense` objects with normalized coordinates, health, and DPS
2. **Data → Graph**: Spatial graph constructed with edges for defenses within connection radius
3. **Graph → Weights**: Strategic weights computed using type-based heuristics and clustering bonuses
4. **Weights + Troops → Optimization**: MILP solver selects optimal troop composition and predicts neutralized defenses
5. **Optimization → UI**: Results displayed with troop counts, neutralized defenses, and visual annotations

---

## Computer Vision Pipeline

### Design Philosophy

BASELINE employs **deterministic, template-based matching** rather than deep learning approaches. This design choice provides:

- **Reproducibility**: Same input always produces same output
- **Interpretability**: Matching scores directly indicate confidence
- **Simplicity**: No training data or model weights required
- **Speed**: Fast inference without GPU acceleration

### Pipeline Stages

#### 1. Preprocessing (`vision/preprocess.py`)

**Input**: Raw RGB image  
**Output**: Grayscale, normalized image

**Operations**:
- Resize to fixed resolution (1024×1024) for scale normalization
- Convert to grayscale (reduces template matching complexity)
- Apply Gaussian blur (σ=0, kernel=5×5) to reduce noise

**Rationale**: Standardization enables consistent template matching across varying screenshot resolutions and quality.

#### 2. Template Matching (`vision/template_matcher.py`)

**Input**: Preprocessed grayscale image  
**Output**: List of detection candidates with coordinates and confidence scores

**Algorithm**: Normalized Cross-Correlation (NCC) using `cv2.matchTemplate` with `TM_CCOEFF_NORMED`

**Mathematical Formulation**:

For template \( T \) and image region \( I \), the normalized cross-correlation coefficient is:

\[
R(x,y) = \frac{\sum_{x',y'} (T(x',y') - \bar{T})(I(x+x',y+y') - \bar{I})}{\sqrt{\sum_{x',y'}(T(x',y') - \bar{T})^2 \sum_{x',y'}(I(x+x',y+y') - \bar{I})^2}}
\]

where \( \bar{T} \) and \( \bar{I} \) are mean values. Values range \([-1, 1]\), with \( R = 1 \) indicating perfect match.

**Thresholding**: Detections with \( R \geq \theta \) (default \( \theta = 0.75 \)) are retained.

**Coordinate Normalization**: Detection centers converted to normalized coordinates \( (x, y) \in [0, 1]^2 \) for resolution independence.

#### 3. Postprocessing (`vision/postprocess.py`)

**Input**: Raw detection list  
**Output**: Deduplicated detection list

**Algorithm**: Greedy distance-based clustering

**Process**:
1. For each detection, find nearest existing cluster (by centroid distance)
2. If distance \( d \leq \tau \) (default \( \tau = 0.05 \)), assign to cluster
3. Otherwise, create new cluster
4. Within each cluster, retain detection with highest confidence score

**Rationale**: Template matching may produce multiple detections for the same physical defense due to:
- Slight template variations
- Overlapping search windows
- Multiple template matches at similar locations

Clustering eliminates duplicates while preserving the highest-confidence detection.

---

## Optimization Formulation

### Mixed-Integer Linear Program

#### Sets

- \( T \): Set of troop types \( i \in \{1, 2, \ldots, m\} \)
- \( D \): Set of defenses \( j \in \{1, 2, \ldots, n\} \)

#### Parameters

**Troop Parameters**:
- \( d_i \): Damage per second (DPS) of troop type \( i \)
- \( h_i \): Health points of troop type \( i \)
- \( c_i \): Housing cost of troop type \( i \)

**Defense Parameters**:
- \( H_j \): Health points of defense \( j \)
- \( D_j \): Damage per second (DPS) of defense \( j \)
- \( w_j \): Strategic weight of defense \( j \)

**Combat Parameters**:
- \( e_{ij} \in [0, 1] \): Effectiveness coefficient of troop \( i \) against defense \( j \)
- \( B \): Housing budget (maximum total housing space)
- \( T \): Estimated combat duration (seconds, for survivability constraint)

#### Decision Variables

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in T
\]

where \( x_i \) is the number of troops of type \( i \) to deploy.

\[
y_j \in \{0, 1\} \quad \forall j \in D
\]

where \( y_j = 1 \) indicates defense \( j \) will be neutralized, and \( y_j = 0 \) otherwise.

#### Objective Function

\[
\max \sum_{j \in D} w_j \cdot y_j
\]

Maximize the weighted sum of neutralized defenses, where weights reflect strategic importance.

#### Constraints

**Housing Budget Constraint**:
\[
\sum_{i \in T} c_i \cdot x_i \leq B
\]

Total housing space used by selected troops cannot exceed available capacity.

**Damage Sufficiency Constraints** (for each defense \( j \)):
\[
\sum_{i \in T} d_i \cdot x_i \cdot e_{ij} \geq H_j \cdot y_j \quad \forall j \in D
\]

If defense \( j \) is to be neutralized (\( y_j = 1 \)), the total effective DPS against it must exceed its health. The effectiveness coefficient \( e_{ij} \) accounts for matchup-specific factors (e.g., air vs. ground, range, armor).

**Survivability Constraint** (optional):
\[
\sum_{i \in T} h_i \cdot x_i \geq \sum_{j \in D} D_j \cdot y_j \cdot T
\]

Total troop health must withstand total defense damage over the estimated combat duration. This constraint ensures the composition can survive long enough to neutralize selected defenses.

#### Strategic Weight Computation

The strategic weight \( w_j \) for defense \( j \) is computed as:

\[
w_j = w_{\text{base}}(t_j) \cdot (1 + \min(\alpha \cdot |N_j|, \beta))
\]

where:
- \( w_{\text{base}}(t_j) \): Base strategic weight for defense type \( t_j \) (heuristic lookup table)
- \( |N_j| \): Number of neighboring defenses within connection radius
- \( \alpha \): Clustering factor (default 0.2)
- \( \beta \): Maximum clustering bonus (default 2.0)

**Rationale**: Defenses in dense clusters create overlapping fire zones and represent strategic chokepoints, warranting higher priority in the objective function.

#### Effectiveness Coefficients

The effectiveness matrix \( e_{ij} \) encodes heuristic matchup values:
- \( e_{ij} = 1.0 \): Maximum effectiveness (e.g., air troops vs. ground-only defenses)
- \( e_{ij} = 0.0 \): No effectiveness (e.g., ground troops vs. air-only defenses)
- \( e_{ij} \in (0, 1) \): Partial effectiveness based on range, armor, and tactical considerations

These coefficients are deterministic heuristics, not learned from data, ensuring interpretability and reproducibility.

#### Solver Implementation

The MILP is solved using **PuLP** with the **CBC** (COIN-OR Branch and Cut) solver backend. The problem structure (integer and binary variables, linear objective, linear constraints) is solved via branch-and-bound with cutting planes.

**Complexity**: The problem is NP-hard in general, but practical instances are tractable due to:
- Moderate problem size (typically \( |T| \leq 10 \), \( |D| < 50 \))
- Sparse constraint structure
- Efficient CBC implementation with preprocessing

---

## Troop Composition Optimization

### Problem Characteristics

The troop composition problem is a **resource allocation MILP** that differs from classical knapsack formulations in several key ways:

1. **Multi-dimensional Resources**: Troops have multiple attributes (DPS, health, housing cost) that interact non-trivially
2. **Combat Interactions**: Effectiveness coefficients \( e_{ij} \) create non-uniform resource-to-target mappings
3. **Combinatorial Structure**: Integer troop counts and binary defense neutralization create a mixed-integer search space
4. **Multiple Constraint Types**: Housing (capacity), damage sufficiency (feasibility), and survivability (safety) constraints

### Solution Interpretation

The optimal solution provides:

1. **Troop Composition**: Integer counts \( x_i^* \) for each troop type, indicating how many of each type to deploy
2. **Neutralized Defenses**: Binary indicators \( y_j^* \) showing which defenses will be destroyed
3. **Resource Utilization**: Total housing used, total DPS output, total health pool
4. **Strategic Value**: Total weighted threat neutralized

### Extensions and Variations

The formulation supports several extensions while maintaining linearity:

- **Multiple Objectives**: Weighted combinations of threat neutralization and resource efficiency
- **Defense Prioritization**: Adjust strategic weights \( w_j \) based on tactical considerations
- **Time-Dependent Constraints**: Vary combat duration \( T \) to model different engagement scenarios
- **Troop Availability**: Add upper bounds \( x_i \leq U_i \) to model limited troop availability

All extensions preserve the MILP structure, ensuring computational tractability.

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies**:
- `opencv-python>=4.8.0`: Computer vision operations
- `numpy>=1.24.0`: Numerical computations
- `networkx>=3.0`: Graph construction and analysis
- `pulp>=2.7.0`: Mixed-integer linear programming
- `streamlit>=1.28.0`: Web application framework
- `matplotlib>=3.7.0`: Visualization
- `Pillow>=10.0.0`: Image manipulation

### Setup

1. Clone or download the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create `templates/` directory and populate with defense template images
4. Run the application: `streamlit run app.py`

### Template Images

Place defense template images in the `templates/` directory. Filenames should match defense types (e.g., `cannon.png`, `archer_tower.jpg`). The filename (without extension) becomes the defense type identifier.

---

## Usage

### Command Line

```bash
python -m streamlit run app.py
```

### Application Workflow

1. **Upload Screenshot**: Select a Builder Base screenshot image
2. **Configure Parameters**: Adjust confidence threshold, deduplication distance, connection radius, housing budget, and survivability settings via sidebar
3. **Review Detections**: Inspect detected defenses with confidence scores, positions, health, and DPS
4. **Manual Editing** (Optional): Add or remove defenses manually to override vision results
5. **View Troop Composition**: Review optimal troop composition (counts per type) and predicted neutralized defenses

### Manual Editing

The system supports human-in-the-loop refinement:

- **Add Defense**: Specify coordinates (normalized [0,1]) and defense type
- **Remove Defense**: Click remove button (🗑️) on any defense in the table
- **Visual Indicators**: Manual defenses marked with "(M)" label and yellow outline

Manual edits persist across image reprocessing and override vision detection results.

---

## Project Structure

```
BASELINE/
├── app.py                      # Streamlit application entry point
├── app/                        # Application module (placeholder)
├── vision/                     # Computer vision pipeline
│   ├── preprocess.py          # Image preprocessing
│   ├── template_matcher.py    # Template matching engine
│   └── postprocess.py         # Detection deduplication
├── data/                       # Data schemas and conversion
│   ├── schema.py              # Defense dataclass and converters
│   └── troops.py              # Troop type definitions and stats
├── graph/                      # Graph construction
│   └── base_graph.py          # NetworkX graph builder
├── optimization/               # Optimization engine
│   ├── weights.py             # Strategic weight computation
│   ├── effectiveness.py       # Troop-defense effectiveness matrix
│   └── solver.py              # MILP formulation and solver
├── templates/                  # Defense template images
│   └── README.md              # Template usage guide
├── config/                     # Configuration (placeholder)
├── utils/                      # Utilities (placeholder)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

### Module Responsibilities

- **`vision/`**: Deterministic computer vision pipeline (preprocessing, template matching, postprocessing)
- **`data/`**: Type-safe data structures (defenses, troops) and conversion utilities
- **`graph/`**: Spatial graph construction from defense positions
- **`optimization/`**: Strategic weight computation, effectiveness modeling, and MILP solver
- **`app.py`**: Streamlit UI orchestration and visualization

---

## Technical Specifications

### Image Processing

- **Input Format**: PNG, JPEG, BMP
- **Processing Resolution**: 1024×1024 pixels (normalized)
- **Color Space**: Grayscale (8-bit)
- **Template Matching**: Normalized Cross-Correlation (TM_CCOEFF_NORMED)
- **Confidence Threshold**: Configurable (default 0.75)

### Graph Construction

- **Graph Type**: Undirected, weighted
- **Edge Weight**: Euclidean distance between defense centers
- **Connection Radius**: Configurable (default 0.2 normalized units)
- **Library**: NetworkX

### Optimization

- **Problem Type**: Resource allocation MILP (mixed-integer linear program)
- **Solver**: PuLP (CBC backend)
- **Variable Types**: Integer (troop counts), Binary (defense neutralization)
- **Objective**: Linear (maximize weighted defense neutralization)
- **Constraints**: Housing capacity, damage sufficiency, survivability (optional)
- **Effectiveness Modeling**: Heuristic coefficients for troop-defense matchups

---

## Limitations and Future Work

### Current Limitations

1. **Template Dependency**: Requires pre-extracted defense templates; does not generalize to unseen defense types
2. **Scale Sensitivity**: Template matching assumes similar scale between templates and screenshots
3. **Heuristic Weights**: Strategic weights and effectiveness coefficients based on domain heuristics rather than learned values
4. **Static Combat Model**: Effectiveness coefficients are static; does not model dynamic combat interactions or timing
5. **Single Time Horizon**: Survivability constraint uses a single estimated combat duration; does not model multi-stage engagements

### Potential Extensions

- **Multi-objective Optimization**: Pareto-optimal solutions considering threat neutralization, resource efficiency, and troop diversity
- **Temporal Planning**: Multi-stage attack planning with dynamic threat updates and troop deployment sequencing
- **Learning-based Parameters**: Train strategic weights and effectiveness coefficients from historical attack success data
- **Adaptive Templates**: Automatic template extraction from annotated examples
- **Dynamic Effectiveness**: Time-varying or context-dependent effectiveness coefficients based on combat state

---

## License

This project is provided as-is for educational and research purposes.

---

## Citation

If you use BASELINE in your research, please cite:

```
BASELINE: Builder Base Strategy & Linear Integer Engine
A prescriptive optimization system for strategic attack planning
using deterministic computer vision and mixed-integer linear programming.
```

---

## Contact

For questions, issues, or contributions, please refer to the project repository.
