# BASELINE - Complete Technical Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technology Stack](#technology-stack)
3. [System Architecture](#system-architecture)
4. [Data Flow Pipeline](#data-flow-pipeline)
5. [Module-by-Module Breakdown](#module-by-module-breakdown)
6. [Key Algorithms & Formulations](#key-algorithms--formulations)
7. [Design Decisions & Rationale](#design-decisions--rationale)
8. [Performance Characteristics](#performance-characteristics)
9. [Extension Points](#extension-points)

---

## Project Overview

**BASELINE** (Builder Base Strategy & Linear Integer Engine) is an end-to-end prescriptive analytics system that transforms raw visual input (screenshots) into optimal resource allocation decisions through a deterministic computer vision pipeline and mixed-integer linear programming.

### Core Problem
Given a base layout screenshot, determine the optimal troop composition that maximizes weighted defense neutralization subject to:
- Housing capacity constraints
- Damage sufficiency requirements
- Survivability guarantees (optional)

### Solution Approach
1. **Computer Vision**: Extract defense positions and types from screenshots
2. **Graph Analysis**: Model spatial relationships between defenses
3. **Optimization**: Solve MILP to find optimal troop composition
4. **Visualization**: Present results in interactive dashboard

---

## Technology Stack

### Core Technologies

#### **Python 3.9+**
- **Role**: Primary programming language
- **Usage**: Entire codebase written in Python
- **Why**: Rich ecosystem for scientific computing, optimization, and web frameworks
- **Key Features Used**:
  - Type hints for code clarity
  - Dataclasses for structured data
  - List/dict comprehensions for data processing

#### **OpenCV (cv2)**
- **Role**: Computer vision and image processing
- **Usage**: 
  - Image preprocessing (`vision/preprocess.py`)
  - Template matching (`vision/template_matcher.py`)
- **Key Functions**:
  - `cv2.imread()` - Load images
  - `cv2.resize()` - Normalize image size
  - `cv2.cvtColor()` - Convert to grayscale
  - `cv2.GaussianBlur()` - Noise reduction
  - `cv2.matchTemplate()` - Template matching with `TM_CCOEFF_NORMED`
- **What It Accomplishes**:
  - Deterministic, reproducible image processing
  - No deep learning dependencies
  - Fast template matching for defense detection
  - Handles varying image resolutions and quality

#### **NumPy**
- **Role**: Numerical computing and array operations
- **Usage**:
  - Image array manipulation
  - Coordinate calculations
  - Mathematical operations in optimization
- **Key Features Used**:
  - `np.ndarray` for image data
  - `np.where()` for finding matches
  - Array indexing and slicing
- **What It Accomplishes**:
  - Efficient numerical operations
  - Memory-efficient image processing
  - Foundation for OpenCV operations

#### **Streamlit**
- **Role**: Web application framework for interactive dashboards
- **Usage**: Main UI in `app.py`
- **Key Components Used**:
  - `st.file_uploader()` - Image upload
  - `st.sidebar` - Configuration panel
  - `st.image()` - Display images
  - `st.dataframe()` - Display tables
  - `st.metric()` - Display KPIs
  - `st.spinner()` - Loading indicators
  - `st.session_state` - State management
- **What It Accomplishes**:
  - Rapid UI development without HTML/CSS/JS
  - Interactive parameter adjustment
  - Real-time visualization updates
  - Session state for manual editing features
  - Professional-looking analytics dashboard

#### **PuLP (Python Linear Programming)**
- **Role**: Mixed-integer linear programming solver interface
- **Usage**: Core optimization in `optimization/solver.py`
- **Key Components**:
  - `pulp.LpProblem()` - Create optimization problem
  - `pulp.LpVariable()` - Define decision variables (Integer, Binary)
  - `pulp.lpSum()` - Linear expressions
  - `pulp.PULP_CBC_CMD()` - Solver backend (CBC)
- **What It Accomplishes**:
  - Formulates MILP problems declaratively
  - Interfaces with CBC solver (COIN-OR Branch and Cut)
  - Handles integer and binary variables
  - Extracts optimal solutions
- **Solver Backend: CBC (COIN-OR Branch and Cut)**
  - Open-source MILP solver
  - Branch-and-bound with cutting planes
  - Efficient for small-to-medium problems
  - Deterministic results

#### **NetworkX**
- **Role**: Graph theory and network analysis
- **Usage**: Spatial graph construction in `graph/base_graph.py`
- **Key Functions**:
  - `nx.Graph()` - Create undirected graph
  - `G.add_node()` - Add defense nodes with attributes
  - `G.add_edge()` - Connect nearby defenses
  - `G.neighbors()` - Find connected defenses
  - `nx.draw_networkx_*()` - Visualization
- **What It Accomplishes**:
  - Models spatial relationships between defenses
  - Enables clustering-based weight calculation
  - Provides graph metrics (density, degree)
  - Supports visualization of defense network

#### **Matplotlib**
- **Role**: Static visualization and plotting
- **Usage**: Graph visualization in `graph/base_graph.py`
- **Key Functions**:
  - `plt.subplots()` - Create figure
  - `nx.draw_networkx_nodes()` - Draw nodes
  - `nx.draw_networkx_edges()` - Draw edges
  - `plt.cm.tab10` - Color mapping
- **What It Accomplishes**:
  - Visual representation of defense graph
  - Color-coded nodes by defense type
  - Publication-quality plots

#### **Pillow (PIL)**
- **Role**: Image manipulation and annotation
- **Usage**: Drawing detection markers in `app.py`
- **Key Classes**:
  - `Image` - Image representation
  - `ImageDraw` - Drawing operations
  - `ImageFont` - Text rendering
- **Key Functions**:
  - `Image.fromarray()` - Convert NumPy to PIL
  - `ImageDraw.Draw()` - Create drawing context
  - `draw.ellipse()` - Draw circles for markers
  - `draw.text()` - Add labels
- **What It Accomplishes**:
  - Annotates images with detection markers
  - Highlights neutralized defenses
  - Adds text labels and priority numbers
  - Converts between image formats

### Supporting Libraries

#### **pathlib.Path**
- **Role**: File system path handling
- **Usage**: Template loading in `vision/template_matcher.py`
- **What It Accomplishes**: Cross-platform path operations, directory iteration

#### **dataclasses**
- **Role**: Structured data definitions
- **Usage**: `Defense`, `TroopType`, `TroopComposition` classes
- **What It Accomplishes**: Type-safe data structures, automatic `__init__`, validation

#### **typing**
- **Role**: Type hints for code documentation
- **Usage**: Function signatures throughout codebase
- **What It Accomplishes**: Better IDE support, self-documenting code, type checking

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                     │
│                    (Streamlit - app.py)                     │
│  - File upload                                              │
│  - Parameter configuration                                 │
│  - Manual editing                                           │
│  - Results visualization                                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                 COMPUTER VISION LAYER                       │
│                    (vision/)                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Preprocessing│→ │   Template   │→ │Postprocessing│     │
│  │  (OpenCV)    │  │   Matching   │  │ (Clustering) │     │
│  │              │  │  (OpenCV)    │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA REPRESENTATION                      │
│                      (data/)                                 │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │   Defense    │  │   Troop      │                         │
│  │   Schema     │  │   Types      │                         │
│  │              │  │              │                         │
│  └──────────────┘  └──────────────┘                         │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    GRAPH ANALYSIS                            │
│                     (graph/)                                │
│  ┌──────────────────────────────────────────────┐          │
│  │  Spatial Graph Construction (NetworkX)        │          │
│  │  - Node: Defense with attributes              │          │
│  │  - Edge: Spatial proximity                    │          │
│  └──────────────────────────────────────────────┘          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  OPTIMIZATION ENGINE                         │
│                  (optimization/)                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Strategic  │  │ Effectiveness│  │    MILP      │     │
│  │   Weights    │  │    Matrix    │  │   Solver     │     │
│  │              │  │              │  │  (PuLP/CBC)  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                            │
│  - Troop composition (counts per type)                      │
│  - Neutralized defenses list                                 │
│  - Visual annotations                                        │
│  - Interactive dashboard                                     │
└─────────────────────────────────────────────────────────────┘
```

### Component Interactions

1. **UI → Vision**: User uploads image, triggers preprocessing
2. **Vision → Data**: Detections converted to typed Defense objects
3. **Data → Graph**: Spatial relationships modeled as NetworkX graph
4. **Graph → Weights**: Strategic weights computed from graph structure
5. **Weights + Troops → Optimization**: MILP formulated and solved
6. **Optimization → UI**: Results displayed with visualizations

---

## Data Flow Pipeline

### Step-by-Step Execution Flow

#### **Phase 1: Image Acquisition & Preprocessing**
```
User uploads screenshot (PNG/JPEG/BMP)
    ↓
app.py: File uploader receives image bytes
    ↓
cv2.imdecode(): Convert bytes to NumPy array (BGR format)
    ↓
preprocess_image_from_array():
    - cv2.resize() → 1024×1024 pixels
    - cv2.cvtColor() → Grayscale
    - cv2.GaussianBlur() → Noise reduction
    ↓
Output: Grayscale NumPy array (1024×1024)
```

#### **Phase 2: Defense Detection**
```
Preprocessed image
    ↓
load_templates(): Scan templates/ directory
    - Load all PNG/JPG images
    - Extract defense type from filename
    - Convert to grayscale
    ↓
detect_defenses():
    For each template:
        cv2.matchTemplate() with TM_CCOEFF_NORMED
        np.where() to find matches above threshold
        Calculate center coordinates
        Normalize to [0,1] range
    ↓
Output: List of detection dicts
    {
        'defense_type': str,
        'x_center': float [0,1],
        'y_center': float [0,1],
        'match_score': float [0,1]
    }
```

#### **Phase 3: Postprocessing**
```
Raw detections list
    ↓
deduplicate_detections():
    - Greedy clustering algorithm
    - Group detections within distance threshold
    - Keep highest-confidence per cluster
    ↓
Output: Deduplicated detection list
```

#### **Phase 4: Data Conversion**
```
Detection dicts
    ↓
detections_to_defenses():
    - Create Defense objects with IDs
    - Lookup default health/DPS from get_defense_stats()
    - Set base_weight = 1.0 (placeholder)
    ↓
Output: List[Defense] objects
    Defense(id, type, x, y, base_weight, health, dps)
```

#### **Phase 5: Graph Construction**
```
Defense objects
    ↓
build_defense_graph():
    - Create NetworkX Graph
    - Add nodes (defenses) with attributes
    - Calculate Euclidean distances
    - Add edges for defenses within radius
    ↓
Output: NetworkX Graph
    - Nodes: Defense IDs with (type, x, y, base_weight)
    - Edges: Connections with weight=distance
```

#### **Phase 6: Strategic Weight Calculation**
```
Graph + Defense objects
    ↓
assign_weights_to_defenses():
    For each defense:
        - Get base weight from DEFENSE_TYPE_WEIGHTS
        - Count neighbors in graph
        - Calculate clustering bonus
        - Final weight = base × (1 + min(0.2 × neighbors, 2.0))
    ↓
Output: Dict[defense_id → strategic_weight]
```

#### **Phase 7: Optimization**
```
Defenses + Strategic Weights + Troop Types
    ↓
solve_troop_composition():
    1. Create PuLP problem (maximize)
    2. Define variables:
       - x[i] ∈ Z≥0 (troop counts)
       - y[j] ∈ {0,1} (defense neutralization)
    3. Add constraints:
       - Housing: Σ c[i]·x[i] ≤ B
       - Damage: Σ d[i]·x[i]·e[i,j] ≥ H[j]·y[j] ∀j
       - Survivability: Σ h[i]·x[i] ≥ Σ D[j]·y[j]·T
    4. Set objective: max Σ w[j]·y[j]
    5. Solve with CBC
    6. Extract solution
    ↓
Output: TroopComposition
    - troop_counts: Dict[str, int]
    - neutralized_defenses: List[Defense]
    - total_housing_used: int
    - total_threat_neutralized: float
    - objective_value: float
```

#### **Phase 8: Visualization**
```
TroopComposition + Original Image
    ↓
app.py visualization:
    - Draw detection markers (PIL)
    - Highlight neutralized defenses (green circles)
    - Display troop composition table
    - Show neutralized defenses table
    - Display metrics (housing used, threat neutralized)
    ↓
Output: Interactive Streamlit dashboard
```

---

## Module-by-Module Breakdown

### **app.py** - Main Application

**Purpose**: Streamlit web application orchestrating entire pipeline

**Key Functions**:
- `preprocess_image_from_array()` - In-memory image preprocessing
- `draw_detections_on_image()` - Annotate image with markers
- `main()` - Main application logic

**Technologies Used**:
- Streamlit: UI framework
- OpenCV: Image loading
- PIL: Image annotation
- NumPy: Array operations

**Session State Management**:
- `manual_defenses`: User-added defenses
- `removed_defense_ids`: User-removed defenses
- `next_manual_id`: ID counter for manual defenses

**UI Components**:
- File uploader
- Sidebar configuration sliders
- Detection visualization
- Manual editing interface
- Results tables and metrics

---

### **vision/preprocess.py** - Image Preprocessing

**Purpose**: Standardize images for template matching

**Key Function**: `preprocess_image()`

**Process**:
1. Load image from file path
2. Resize to 1024×1024 (normalization)
3. Convert BGR → Grayscale
4. Apply Gaussian blur (5×5 kernel)

**Technologies**:
- OpenCV: `cv2.imread()`, `cv2.resize()`, `cv2.cvtColor()`, `cv2.GaussianBlur()`

**Why These Steps**:
- **Resize**: Ensures consistent scale for template matching
- **Grayscale**: Reduces complexity, template matching works on grayscale
- **Blur**: Reduces noise, improves matching robustness

---

### **vision/template_matcher.py** - Defense Detection

**Purpose**: Detect defense structures using template matching

**Key Functions**:
- `load_templates()` - Load template images from directory
- `match_template()` - Match single template against image
- `detect_defenses()` - Main detection function

**Algorithm**: Normalized Cross-Correlation (NCC)

**Mathematical Formulation**:
```
R(x,y) = Σ(T(x',y') - T̄)(I(x+x',y+y') - Ī) / 
         √[Σ(T(x',y') - T̄)² · Σ(I(x+x',y+y') - Ī)²]
```

**Technologies**:
- OpenCV: `cv2.matchTemplate()` with `TM_CCOEFF_NORMED`
- NumPy: `np.where()` for finding matches
- pathlib: Directory traversal

**Parameters**:
- `confidence_threshold`: Minimum match score (default 0.75)
- `templates_dir`: Directory containing template images

**Output Format**:
```python
[
    {
        'defense_type': 'cannon',
        'x_center': 0.45,  # Normalized [0,1]
        'y_center': 0.32,
        'match_score': 0.87
    },
    ...
]
```

---

### **vision/postprocess.py** - Detection Deduplication

**Purpose**: Remove duplicate detections from template matching

**Key Function**: `deduplicate_detections()`

**Algorithm**: Greedy distance-based clustering

**Process**:
1. For each detection, find nearest existing cluster
2. If distance ≤ threshold, add to cluster
3. Otherwise, create new cluster
4. Within each cluster, keep highest-confidence detection

**Technologies**:
- Pure Python: No external libraries
- Math: Euclidean distance calculation

**Parameters**:
- `distance_threshold`: Maximum distance for clustering (default 0.05)

**Why Needed**: Template matching can produce multiple detections for same physical defense due to:
- Overlapping search windows
- Slight template variations
- Multiple template matches

---

### **data/schema.py** - Defense Data Model

**Purpose**: Define defense data structure and conversion utilities

**Key Classes**:
- `Defense`: Dataclass representing a defense structure

**Defense Attributes**:
- `id`: Unique identifier
- `type`: Defense type name
- `x`, `y`: Normalized coordinates [0,1]
- `base_weight`: Strategic importance
- `health`: Hit points
- `dps`: Damage per second

**Key Functions**:
- `detections_to_defenses()`: Convert vision detections to Defense objects
- `get_defense_stats()`: Lookup default health/DPS by type
- `defense_to_dict()`: Serialize to dictionary

**Technologies**:
- `dataclasses`: Structured data definition
- `typing`: Type hints

**Default Defense Stats** (examples):
- `crusher`: (2000 HP, 150 DPS)
- `giant_cannon`: (1800 HP, 120 DPS)
- `cannon`: (1000 HP, 50 DPS)

---

### **data/troops.py** - Troop Data Model

**Purpose**: Define troop types and their combat statistics

**Key Classes**:
- `TroopType`: Dataclass representing a troop type

**Troop Attributes**:
- `name`: Troop type name
- `dps`: Damage per second
- `health`: Hit points
- `housing_cost`: Housing space required

**Troop Database**: `TROOP_TYPES` dictionary with 10 troop types

**Examples**:
- `barbarian`: 25 DPS, 150 HP, 1 housing
- `giant`: 30 DPS, 500 HP, 3 housing
- `super_pekka`: 100 DPS, 800 HP, 8 housing

**Key Functions**:
- `get_troop_type()`: Lookup by name
- `get_all_troop_types()`: Get all troops
- `get_troop_names()`: Get list of names

**Technologies**:
- `dataclasses`: Structured data
- `typing`: Type hints

---

### **graph/base_graph.py** - Spatial Graph Construction

**Purpose**: Model spatial relationships between defenses

**Key Functions**:
- `euclidean_distance()`: Calculate distance between defenses
- `build_defense_graph()`: Create NetworkX graph
- `visualize_graph()`: Matplotlib visualization
- `get_graph_stats()`: Graph metrics

**Graph Structure**:
- **Nodes**: Defense IDs with attributes (type, x, y, base_weight)
- **Edges**: Connections between defenses within `connection_radius`
- **Edge Weight**: Euclidean distance

**Technologies**:
- NetworkX: Graph construction and analysis
- Matplotlib: Visualization
- Math: Euclidean distance calculation

**Parameters**:
- `connection_radius`: Maximum distance for connection (default 0.2)

**Use Case**: Enables clustering-based weight calculation (defenses with many neighbors get higher strategic weights)

---

### **optimization/weights.py** - Strategic Weight Calculation

**Purpose**: Compute strategic importance weights for defenses

**Key Functions**:
- `get_base_weight()`: Lookup base weight by defense type
- `calculate_clustering_bonus()`: Compute neighbor-based bonus
- `calculate_threat_score()`: Combine base weight and clustering
- `assign_weights_to_defenses()`: Main weight assignment function

**Weight Formula**:
```
w[i] = w_base(type[i]) × (1 + min(0.2 × neighbors[i], 2.0))
```

**Base Weights** (examples):
- `crusher`: 10.0
- `giant_cannon`: 9.0
- `cannon`: 5.5
- `wall`: 0.5

**Clustering Logic**:
- Defenses with more neighbors get higher weights
- Represents overlapping fire zones and strategic chokepoints
- Bonus capped at 2.0× multiplier

**Technologies**:
- NetworkX: Graph neighbor counting
- Pure Python: Mathematical calculations

---

### **optimization/effectiveness.py** - Combat Effectiveness Matrix

**Purpose**: Define heuristic effectiveness coefficients for troop-defense matchups

**Key Data Structure**: `EFFECTIVENESS_MATRIX`

**Format**: `Dict[Tuple[troop_type, defense_type] → float]`

**Value Range**: [0, 1]
- `1.0`: Maximum effectiveness (e.g., air troops vs ground-only defenses)
- `0.0`: No effectiveness
- `0.5`: Default for unknown matchups

**Examples**:
- `('minion', 'crusher') = 1.0` (air vs ground-only)
- `('archer', 'crusher') = 0.9` (range advantage)
- `('barbarian', 'crusher') = 0.3` (melee disadvantage)

**Key Function**: `get_effectiveness()` - Lookup effectiveness coefficient

**Technologies**:
- Pure Python: Dictionary lookup
- `typing`: Type hints

**Design Philosophy**: Deterministic heuristics, not physics-based or learned from data

---

### **optimization/solver.py** - MILP Optimization Engine

**Purpose**: Solve troop composition optimization problem

**Key Function**: `solve_troop_composition()`

**Mathematical Formulation**:

**Sets**:
- `T`: Troop types
- `D`: Defenses

**Decision Variables**:
- `x[i] ∈ Z≥0`: Number of troops of type i
- `y[j] ∈ {0,1}`: Whether defense j is neutralized

**Objective**:
```
maximize: Σ w[j] · y[j]
```

**Constraints**:

1. **Housing Budget**:
   ```
   Σ c[i] · x[i] ≤ B
   ```

2. **Damage Sufficiency** (for each defense j):
   ```
   Σ d[i] · x[i] · e[i,j] ≥ H[j] · y[j]
   ```
   Ensures if defense is neutralized (y[j]=1), sufficient DPS is allocated.

3. **Survivability** (optional):
   ```
   Σ h[i] · x[i] ≥ Σ D[j] · y[j] · T
   ```
   Ensures total troop health can withstand defense damage over time T.

**Technologies**:
- PuLP: Problem formulation
- CBC: Solver backend (via PuLP)
- NumPy: Numerical operations (implicit)

**Solution Extraction**:
- Extract integer troop counts from `x[i]` variables
- Extract binary neutralization indicators from `y[j]` variables
- Calculate totals (housing used, threat neutralized)

**Return Type**: `TroopComposition` dataclass

---

## Key Algorithms & Formulations

### 1. Normalized Cross-Correlation (Template Matching)

**Purpose**: Find template matches in image

**Formula**:
```
R(x,y) = Σ(T(x',y') - T̄)(I(x+x',y+y') - Ī) / 
         √[Σ(T(x',y') - T̄)² · Σ(I(x+x',y+y') - Ī)²]
```

**Implementation**: OpenCV `cv2.matchTemplate()` with `TM_CCOEFF_NORMED`

**Properties**:
- Normalized to [-1, 1] range
- `R = 1`: Perfect match
- `R = -1`: Perfect inverse match
- Invariant to linear brightness changes

**Threshold**: Default 0.75 (75% match confidence)

---

### 2. Greedy Distance-Based Clustering

**Purpose**: Remove duplicate detections

**Algorithm**:
```
clusters = []
for detection in detections:
    nearest_cluster = find_nearest_cluster(detection, clusters)
    if distance(detection, nearest_cluster) ≤ threshold:
        add_to_cluster(detection, nearest_cluster)
    else:
        create_new_cluster(detection)
    
for cluster in clusters:
    keep_highest_confidence(cluster)
```

**Distance Metric**: Euclidean distance in normalized coordinate space

**Time Complexity**: O(n²) worst case, but typically faster due to early termination

---

### 3. Strategic Weight Calculation

**Formula**:
```
w[i] = w_base(type[i]) × (1 + min(α × |N[i]|, β))
```

Where:
- `w_base(type[i])`: Base weight from lookup table
- `|N[i]|`: Number of neighboring defenses
- `α`: Clustering factor (default 0.2)
- `β`: Maximum bonus (default 2.0)

**Rationale**: 
- Base weight reflects inherent threat level
- Clustering bonus reflects strategic importance of dense defense clusters
- Capped to prevent extreme values

---

### 4. Mixed-Integer Linear Programming

**Problem Type**: Resource allocation MILP

**Complexity**: NP-hard in general, but tractable for small instances

**Solver**: Branch-and-bound with cutting planes (CBC)

**Variable Types**:
- Integer: Troop counts (can be 0, 1, 2, ...)
- Binary: Defense neutralization (0 or 1)

**Constraint Types**:
- Linear inequalities
- All coefficients are constants (no nonlinear terms)

**Solution Method**:
1. Relax integer constraints → Linear Program (LP)
2. Solve LP relaxation
3. If solution is integer, done
4. Otherwise, branch on fractional variable
5. Repeat recursively with bounds

---

## Design Decisions & Rationale

### Why Deterministic Computer Vision?

**Decision**: Use OpenCV template matching instead of deep learning

**Rationale**:
- **Reproducibility**: Same input → same output
- **Interpretability**: Match scores directly indicate confidence
- **Simplicity**: No training data or model weights
- **Speed**: Fast inference without GPU
- **Transparency**: Easy to debug and understand

**Trade-offs**:
- Less robust to variations in appearance
- Requires manual template creation
- May miss novel defense types

---

### Why Mixed-Integer Linear Programming?

**Decision**: Use MILP instead of heuristics or simulation

**Rationale**:
- **Optimality**: Guarantees optimal solution (within solver tolerance)
- **Mathematical Rigor**: Well-defined problem formulation
- **Constraint Handling**: Natural way to express resource limits
- **Extensibility**: Easy to add new constraints
- **Deterministic**: Same problem → same solution

**Trade-offs**:
- Requires linear constraints (no nonlinear relationships)
- Solver time increases with problem size
- Integer variables make problem harder than LP

---

### Why Graph-Based Weight Calculation?

**Decision**: Use spatial graph to adjust strategic weights

**Rationale**:
- **Spatial Context**: Defenses in clusters are more strategically important
- **Overlapping Fire Zones**: Dense clusters create dangerous areas
- **Chokepoints**: Clustered defenses represent key tactical positions
- **Simple Heuristic**: Easy to understand and tune

**Trade-offs**:
- Heuristic nature (not learned from data)
- Connection radius is a hyperparameter
- May not capture all strategic nuances

---

### Why Effectiveness Coefficients?

**Decision**: Use static effectiveness matrix instead of dynamic calculation

**Rationale**:
- **Linearity**: Keeps MILP constraints linear
- **Interpretability**: Clear matchup values
- **Simplicity**: Easy to understand and modify
- **Deterministic**: No randomness or uncertainty

**Trade-offs**:
- Static values don't adapt to context
- May not capture all combat nuances
- Requires domain expertise to set values

---

## Performance Characteristics

### Computational Complexity

**Template Matching**:
- Time: O(W × H × T_w × T_h) per template
  - W, H: Image dimensions
  - T_w, T_h: Template dimensions
- Space: O(W × H) for result matrix

**Clustering**:
- Time: O(n²) worst case, O(n·k) average (k = clusters)
- Space: O(n)

**Graph Construction**:
- Time: O(n²) for distance calculations
- Space: O(n + m) where m = edges

**MILP Solving**:
- Time: Exponential worst case, but typically fast for small problems
- Space: O(n + m) for problem representation
- Typical problem size: < 50 defenses, < 10 troop types

### Memory Usage

- Image processing: ~4-8 MB per image
- Graph: ~1-10 KB depending on defense count
- MILP problem: ~10-100 KB
- Total: < 50 MB for typical use case

### Scalability

**Current Limits**:
- Handles up to ~100 defenses comfortably
- Up to ~20 troop types
- Solver time: < 5 seconds for typical problems

**Bottlenecks**:
- Template matching: Linear in number of templates
- Graph construction: Quadratic in number of defenses
- MILP solving: Exponential worst case

**Optimization Opportunities**:
- Parallel template matching
- Spatial indexing for graph construction
- Problem decomposition for large instances

---

## Extension Points

### Adding New Troop Types

1. Add entry to `data/troops.py`:
```python
TROOP_TYPES['new_troop'] = TroopType(
    name='new_troop',
    dps=40.0,
    health=200.0,
    housing_cost=2
)
```

2. Add effectiveness coefficients to `optimization/effectiveness.py`:
```python
EFFECTIVENESS_MATRIX[('new_troop', 'cannon')] = 0.8
# ... for all defense types
```

### Adding New Defense Types

1. Add base weight to `optimization/weights.py`:
```python
DEFENSE_TYPE_WEIGHTS['new_defense'] = 6.0
```

2. Add default stats to `data/schema.py`:
```python
stats = {
    'new_defense': (1200.0, 60.0),  # (health, dps)
    ...
}
```

3. Create template image: `templates/new_defense.png`

### Adding New Constraints

Modify `optimization/solver.py` in `solve_troop_composition()`:

```python
# Example: Maximum troops per type
for troop in troop_types:
    max_troops_constraint = troop_vars[troop.name] <= max_per_type
    problem += max_troops_constraint, f"Max_Troops_{troop.name}"
```

### Custom Objective Functions

Modify objective in `solve_troop_composition()`:

```python
# Example: Minimize housing while maximizing threat
objective = (
    pulp.lpSum([...]) -  # Maximize threat
    0.1 * pulp.lpSum([...])  # Minimize housing (weighted)
)
```

---

## Summary

**BASELINE** is a complete prescriptive analytics system demonstrating:

1. **End-to-End Pipeline**: From raw images to optimization results
2. **Deterministic Processing**: Reproducible, interpretable results
3. **Mathematical Rigor**: MILP formulation with proven optimality
4. **Modular Architecture**: Clean separation of concerns
5. **Production-Ready Code**: Type hints, error handling, documentation

**Key Technologies**:
- **Python**: Core language
- **OpenCV**: Computer vision
- **Streamlit**: Web UI
- **PuLP/CBC**: Optimization
- **NetworkX**: Graph analysis
- **NumPy/PIL**: Data processing

**Use Cases**:
- Prescriptive analytics demonstrations
- Resource allocation optimization
- Computer vision applications
- MILP problem solving
- Interactive dashboard development

This system showcases skills in:
- Computer vision and image processing
- Mathematical optimization
- Software engineering and architecture
- Data pipeline design
- Web application development

