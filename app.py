"""
BASELINE - Builder Base Strategy & Linear Integer Engine

Streamlit application for analyzing Clash of Clans Builder Base screenshots
and generating optimal attack strategies.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List

# Import BASELINE modules
from vision import preprocess_image, detect_defenses, deduplicate_detections
from data import detections_to_defenses, Defense, get_all_troop_types
from graph import build_defense_graph, visualize_graph, get_graph_stats
from optimization import solve_troop_composition_with_weights, print_troop_composition


# Page configuration
st.set_page_config(
    page_title="BASELINE - Builder Base Analyzer",
    page_icon="⚔️",
    layout="wide"
)

# Title
st.title("⚔️ BASELINE - Builder Base Strategy Engine")
st.markdown("Upload a Builder Base screenshot to get an optimized attack strategy.")

# Configuration parameters (default values)
confidence_threshold = 0.30  # Lowered significantly based on diagnostic results
distance_threshold = 0.05
connection_radius = 0.2
max_army_camps = 6
use_survivability = True
time_estimate = 30.0
templates_dir = "templates"


def preprocess_image_from_array(
    image: np.ndarray,
    target_size: tuple[int, int] = (1024, 1024),
    blur_kernel_size: int = 5,
    preserve_aspect: bool = False
) -> np.ndarray:
    """
    Preprocess an image from a numpy array (in-memory).
    
    Args:
        image: Input image as numpy array (BGR format)
        target_size: Target resolution as (width, height). Default: (1024, 1024)
        blur_kernel_size: Size of Gaussian blur kernel (must be odd). Default: 5
        preserve_aspect: If True, resize maintaining aspect ratio and pad to target size. Default: False
    
    Returns:
        Preprocessed grayscale image as NumPy array
    """
    if blur_kernel_size % 2 == 0:
        raise ValueError(f"blur_kernel_size must be odd, got {blur_kernel_size}")
    
    if preserve_aspect:
        # Resize maintaining aspect ratio
        original_height, original_width = image.shape[:2]
        target_width, target_height = target_size
        
        # Calculate scaling factor
        scale = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize
        image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size (centered, black borders)
        pad_top = (target_height - new_height) // 2
        pad_bottom = target_height - new_height - pad_top
        pad_left = (target_width - new_width) // 2
        pad_right = target_width - new_width - pad_left
        
        image_resized = cv2.copyMakeBorder(
            image_resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    else:
        # Resize to target resolution (may distort aspect ratio)
        image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Convert to grayscale
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    image_blurred = cv2.GaussianBlur(
        image_gray,
        (blur_kernel_size, blur_kernel_size),
        sigmaX=0
    )
    
    return image_blurred


def draw_detections_on_image(image: np.ndarray, detections: List[dict]) -> np.ndarray:
    """
    Draw detection markers on the image.
    
    Args:
        image: Input image (BGR format from OpenCV)
        detections: List of detection dictionaries with x_center, y_center, defense_type
    
    Returns:
        Image with detection markers drawn
    """
    # Convert BGR to RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    # Get image dimensions
    img_width, img_height = pil_image.size
    
    # Color mapping for different defense types
    colors = {
        'cannon': (255, 0, 0),      # Red
        'archer_tower': (0, 255, 0), # Green
        'crusher': (0, 0, 255),      # Blue
        'giant_cannon': (255, 165, 0), # Orange
        'multi_mortar': (255, 0, 255), # Magenta
        'default': (255, 255, 0)     # Yellow
    }
    
    # Draw each detection
    for detection in detections:
        x_norm = detection['x_center']
        y_norm = detection['y_center']
        defense_type = detection['defense_type']
        
        # Convert normalized coordinates to pixel coordinates
        x_px = int(x_norm * img_width)
        y_px = int(y_norm * img_height)
        
        # Get color for this defense type
        color = colors.get(defense_type, colors['default'])
        
        # Draw circle marker
        radius = 15
        outline_width = 2
        outline_color = (255, 255, 255)
        
        draw.ellipse(
            [x_px - radius, y_px - radius, x_px + radius, y_px + radius],
            fill=color,
            outline=outline_color,
            width=outline_width
        )
        
        # Draw defense type label
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        draw.text(
            (x_px + radius + 5, y_px - 10),
            defense_type,
            fill=color,
            font=font
        )
    
    # Convert back to numpy array
    return np.array(pil_image)


def main():
    """Main application logic."""
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Builder Base Screenshot",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Upload a screenshot of a Clash of Clans Builder Base"
    )
    
    if uploaded_file is None:
        st.info("👆 Please upload a Builder Base screenshot to begin analysis.")
        return
    
    # Load image
    file_bytes = uploaded_file.read()
    image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    if image is None:
        st.error("Failed to load image. Please try a different file.")
        return
    
    # Processing pipeline
    with st.spinner("Processing image and detecting defenses..."):
        try:
            # Step 1: Preprocessing
            original_height, original_width = image.shape[:2]
            
            # Try with aspect ratio preservation first
            processed_image = preprocess_image_from_array(
                image,
                target_size=(1024, 1024),
                blur_kernel_size=5,
                preserve_aspect=True
            )
            
            # Step 2: Template matching
            try:
                from vision.template_matcher import load_templates
                templates = load_templates(templates_dir)
            except Exception as e:
                st.error(f"Failed to load templates: {str(e)}")
                return
            
            raw_detections = detect_defenses(
                processed_image,
                templates_dir=templates_dir,
                confidence_threshold=confidence_threshold,
                use_multi_scale=True  # Enable multi-scale matching for better accuracy
            )
            
            # If no detections with default threshold, try lower threshold
            if not raw_detections:
                raw_detections = detect_defenses(
                    processed_image,
                    templates_dir=templates_dir,
                    confidence_threshold=0.30,
                    use_multi_scale=True  # Enable multi-scale matching
                )
            
            if not raw_detections:
                st.error("No defenses detected. Please ensure your image is a Builder Base screenshot.")
                return
            
            # Step 3: Postprocessing (deduplication)
            unique_detections = deduplicate_detections(
                raw_detections,
                distance_threshold=distance_threshold,
                same_type_only=True  # Only cluster detections of the same defense type
            )
            
            if not unique_detections:
                st.error("No defenses detected after processing. Please try a different image.")
                return
            
            # Step 4: Convert to Defense objects
            defenses = detections_to_defenses(unique_detections, start_id=0)
            
            # Create detection dicts for visualization
            all_detections = []
            for detection in unique_detections:
                all_detections.append({
                    'defense_type': detection['defense_type'],
                    'x_center': detection['x_center'],
                    'y_center': detection['y_center'],
                    'match_score': detection.get('match_score', 1.0)
                })
            
            # Step 5: Graph construction
            with st.spinner("Analyzing base layout..."):
                G = build_defense_graph(defenses, connection_radius=connection_radius)
            
            # Step 6: Optimization
            with st.spinner("Calculating optimal troop composition..."):
                composition = solve_troop_composition_with_weights(
                    defenses=defenses,
                    max_army_camps=max_army_camps,
                    graph=G,
                    base_clustering_factor=0.2,
                    max_bonus=2.0,
                    use_survivability=use_survivability,
                    time_estimate=time_estimate
                )
            
            # Display results in two columns: image on left, composition on right
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.subheader("Uploaded Base")
                st.image(image, channels="BGR", caption="Builder Base Screenshot", use_container_width=True)
            
            with col_right:
                st.subheader("🎯 Optimal Troop Composition")
                
                if not composition.troop_counts and not composition.hero_type:
                    st.warning("No troop composition generated. Try adjusting the constraints.")
                else:
                    # Summary metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Troop Types Selected", len(composition.troop_counts))
                        st.metric("Defenses Neutralized", len(composition.neutralized_defenses))
                    with col2:
                        st.metric("Threat Neutralized", f"{composition.total_threat_neutralized:.2f}")
                        st.metric("Objective Value", f"{composition.objective_value:.2f}")
                    
                    # Troop composition
                    st.markdown("### Recommended Troop Composition")
                    
                    # Display army camps
                    st.markdown("#### Army Camps (6 camps, each with 1 troop type)")
                    troop_data = []
                    total_troops = 0
                    for troop_name, count in sorted(composition.troop_counts.items()):
                        from data.troops import get_troop_type
                        troop = get_troop_type(troop_name)
                        total_troops += count
                        troop_data.append({
                            "Camp": f"Camp {len(troop_data) + 1}",
                            "Troop Type": troop_name.capitalize(),
                            "Count": count,
                            "DPS": f"{troop.dps:.1f}",
                            "Health": f"{troop.health:.1f}",
                            "Total DPS": f"{troop.dps * count:.1f}",
                            "Total Health": f"{troop.health * count:.1f}"
                        })
                    
                    if troop_data:
                        st.dataframe(troop_data, use_container_width=True)
                        st.caption(f"Total Troops: {total_troops} | Troop Types Used: {len(composition.troop_counts)}/{max_army_camps}")
                    else:
                        st.info("No troops selected.")
                    
                    # Display hero
                    st.markdown("#### Hero Camp (1 hero)")
                    if composition.hero_type:
                        from data.troops import get_hero_type
                        hero = get_hero_type(composition.hero_type)
                        hero_data = [{
                            "Hero Type": composition.hero_type.replace('_', ' ').title(),
                            "DPS": f"{hero.dps:.1f}",
                            "Health": f"{hero.health:.1f}"
                        }]
                        st.dataframe(hero_data, use_container_width=True)
                    else:
                        st.info("No hero selected.")
                    
                    # Neutralized defenses
                    st.markdown("### Predicted Neutralized Defenses")
                    
                    if composition.neutralized_defenses:
                        defense_data = []
                        for defense in composition.neutralized_defenses:
                            defense_data.append({
                                "ID": defense.id,
                                "Type": defense.type,
                                "Health": f"{defense.health:.1f}",
                                "DPS": f"{defense.dps:.1f}",
                                "Weight": f"{defense.base_weight:.2f}",
                                "Position": f"({defense.x:.3f}, {defense.y:.3f})"
                            })
                        st.dataframe(defense_data, use_container_width=True)
                    else:
                        st.info("No defenses will be neutralized with the current troop composition.")
                
        
        except FileNotFoundError as e:
            st.error(f"Templates directory not found: {templates_dir}")
            st.info("Please ensure template images are placed in the templates directory.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()

