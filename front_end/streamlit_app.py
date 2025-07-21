import streamlit as st
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from model_utils import SegmentationModel
import os

# Page configuration
st.set_page_config(
    page_title="Segmentación de Plantas de Maíz",
    page_icon="🌽",
    layout="wide"
)

# Initialize session state
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'a106acdd252ec8c131d81b70a2014ffc'

if 'model' not in st.session_state:
    with st.spinner("Cargando modelo..."):
        st.session_state.model = SegmentationModel(st.session_state.selected_model)

def apply_threshold(prediction, threshold):
    """Apply threshold to prediction"""
    return (prediction >= threshold).astype(np.uint8)

def create_heatmap(prediction):
    """Create heatmap visualization"""
    # Normalize prediction to 0-1 range
    normalized = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
    
    # Apply colormap
    colormap = cm.get_cmap('jet')
    heatmap = colormap(normalized)
    
    # Convert to RGB and scale to 0-255
    heatmap_rgb = (heatmap[:, :, :3] * 255).astype(np.uint8)
    
    return heatmap_rgb

def overlay_mask_on_image(image, mask, alpha=0.5):
    """Overlay mask on original image"""
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Create colored mask (green)
    colored_mask = np.zeros_like(img_array)
    colored_mask[:, :, 1] = mask * 255  # Green channel
    
    # Blend
    overlay = cv2.addWeighted(img_array, 1-alpha, colored_mask, alpha, 0)
    
    return overlay

# Main UI
st.title("🌽 Segmentación de Plantas de Maíz")
st.markdown("Sistema de inferencia para segmentación binaria de plantas")

# Sidebar for controls
st.sidebar.header("Controles")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Seleccionar imagen",
    type=['jpg', 'jpeg', 'png'],
    help="Sube una imagen para realizar la inferencia"
)

# Inference method selection
inference_method = st.sidebar.radio(
    "Método de inferencia:",
    ["full", "bottom_half"],
    format_func=lambda x: "Imagen completa" if x == "full" else "Mitad inferior"
)

# Threshold slider
threshold = st.sidebar.slider(
    "Umbral de predicción",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Ajusta el umbral para la binarización de la máscara"
)

# Display options
show_heatmap = st.sidebar.checkbox("Mostrar mapa de calor", value=True)
show_overlay = st.sidebar.checkbox("Mostrar superposición", value=True)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Display original image info
    st.subheader("Imagen Original")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.image(image, caption=f"Imagen cargada: {uploaded_file.name}", width=400)
    
    with col2:
        st.write(f"**Resolución:** {image.size[0]} x {image.size[1]}")
        st.write(f"**Método:** {'Imagen completa' if inference_method == 'full' else 'Mitad inferior'}")
        st.write(f"**Umbral:** {threshold:.2f}")
    
    with col3:
        st.subheader("Selección de Modelo")
        
        model_options = {
            '3f0ad160098a0f90e049b0ddc0b4dc6e': 'IMAGEN COMPLETA',
            '8f7844d98e8d29e93b3831b9576a7db4': 'MITAD DE IMAGEN', 
            'a106acdd252ec8c131d81b70a2014ffc': 'ESTADIO FENOLICO BAJO',
            'e82fb89e7ef3ff71c7fe00621b4c7bed': 'ESTADIO FENOLICO ALTO'
        }
        
        selected_model = st.selectbox(
            "Especializado en:",
            options=list(model_options.keys()),
            index=list(model_options.keys()).index(st.session_state.selected_model),
            format_func=lambda x: model_options[x]
        )
        
        # Check if model changed
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            with st.spinner("Cambiando modelo..."):
                st.session_state.model = SegmentationModel(selected_model)
            st.rerun()
    
    # Predict
    with st.spinner("Ejecutando inferencia..."):
        prediction = st.session_state.model.predict_image(image, inference_method)
    
    # Apply threshold
    binary_mask = apply_threshold(prediction, threshold)
    
    # Count objects
    object_count, labels, centroids = st.session_state.model.count_objects(binary_mask)
    
    # Display results
    st.subheader("Resultados")
    
    # Metrics row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Objetos detectados", object_count)
    
    with col2:
        st.metric("Píxeles positivos", np.sum(binary_mask))
    
    with col3:
        coverage_percentage = (np.sum(binary_mask) / binary_mask.size) * 100
        st.metric("Cobertura", f"{coverage_percentage:.1f}%")
    
    # Visualizations
    viz_cols = st.columns(2 if show_heatmap else 1)
    
    with viz_cols[0]:
        st.subheader("Máscara de Segmentación")
        
        if show_overlay:
            overlay = overlay_mask_on_image(image, binary_mask)
            st.image(overlay, caption="Máscara superpuesta", use_container_width=True)
        else:
            # Show binary mask
            mask_colored = np.stack([binary_mask * 255] * 3, axis=-1)
            st.image(mask_colored, caption="Máscara binaria", use_container_width=True)
    
    if show_heatmap and len(viz_cols) > 1:
        with viz_cols[1]:
            st.subheader("Mapa de Calor")
            heatmap = create_heatmap(prediction)
            st.image(heatmap, caption="Probabilidades de predicción", use_container_width=True)
    
    # Object detection visualization
    if object_count > 0:
        st.subheader("Detección de Objetos")
        
        # Create visualization with centroids
        img_with_centroids = np.array(image).copy()
        
        # Draw centroids
        for i, (x, y) in enumerate(centroids):
            cv2.circle(img_with_centroids, (int(x), int(y)), 5, (255, 0, 0), -1)
            cv2.putText(img_with_centroids, str(i+1), (int(x)+10, int(y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        st.image(img_with_centroids, caption=f"Centroides de {object_count} objetos detectados", use_container_width=True)
        
        # Show centroids table
        if st.checkbox("Mostrar tabla de centroides"):
            centroid_data = []
            for i, (x, y) in enumerate(centroids):
                centroid_data.append({
                    "Objeto": i+1,
                    "X": f"{x:.1f}",
                    "Y": f"{y:.1f}"
                })
            st.table(centroid_data)
    
    # Point mask validation section
    st.subheader("Validación con Máscara de Puntos")
    st.markdown("Sube una máscara binaria de puntos para calcular métricas de posicionamiento")
    
    uploaded_point_mask = st.file_uploader(
        "Seleccionar máscara de puntos binaria",
        type=['jpg', 'jpeg', 'png'],
        help="Sube una imagen binaria con puntos marcados para validar la predicción",
        key="point_mask_uploader"
    )
    
    if uploaded_point_mask is not None:
        # Load and process point mask
        point_mask_pil = Image.open(uploaded_point_mask).convert('L')  # Convert to grayscale
        point_mask = np.array(point_mask_pil)
        
        # Resize point mask to match prediction size
        if point_mask.shape != binary_mask.shape:
            point_mask = cv2.resize(point_mask, (binary_mask.shape[1], binary_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Binarize point mask (threshold at 127)
        point_mask_binary = (point_mask > 127).astype(np.uint8)
        
        # Find point coordinates (non-zero pixels in point mask)
        point_coords = np.column_stack(np.where(point_mask_binary > 0))
        
        # Calculate how many points fall outside predicted regions
        points_outside = 0
        points_inside = 0
        
        for y, x in point_coords:
            if binary_mask[y, x] > 0:
                points_inside += 1
            else:
                points_outside += 1
        
        total_points = len(point_coords)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Puntos totales", total_points)
        
        with col2:
            st.metric("Puntos dentro", points_inside)
        
        with col3:
            st.metric("Puntos fuera", points_outside)
        
        with col4:
            accuracy_percentage = (points_inside / total_points) * 100 if total_points > 0 else 0
            st.metric("Precisión", f"{accuracy_percentage:.1f}%")
        
        # Create visualization with overlay
        overlay_viz_cols = st.columns(2)
        
        with overlay_viz_cols[0]:
            st.subheader("Superposición de Máscaras")
            
            # Create RGB visualization
            overlay_img = np.array(image).copy()
            
            # Add predicted mask (green, semi-transparent)
            mask_overlay = np.zeros_like(overlay_img)
            mask_overlay[:, :, 1] = binary_mask * 255  # Green for predictions
            overlay_img = cv2.addWeighted(overlay_img, 0.7, mask_overlay, 0.3, 0)
            
            # Add points (red for outside, blue for inside)
            for y, x in point_coords:
                color = (255, 0, 0) if binary_mask[y, x] == 0 else (0, 0, 255)  # Red if outside, blue if inside
                cv2.circle(overlay_img, (x, y), 3, color, -1)
            
            st.image(overlay_img, caption="Verde: Predicción | Rojo: Puntos fuera | Azul: Puntos dentro", use_container_width=True)
        
        with overlay_viz_cols[1]:
            st.subheader("Máscara de Puntos Original")
            
            # Show original point mask
            point_mask_colored = np.stack([point_mask] * 3, axis=-1)
            st.image(point_mask_colored, caption="Máscara de puntos cargada", use_container_width=True)
        
        # Detailed analysis
        if st.checkbox("Mostrar análisis detallado"):
            st.subheader("Análisis Detallado")
            
            # Show point coordinates table
            if st.checkbox("Mostrar coordenadas de puntos"):
                point_data = []
                for i, (y, x) in enumerate(point_coords):
                    status = "Dentro" if binary_mask[y, x] > 0 else "Fuera"
                    point_data.append({
                        "Punto": i+1,
                        "X": x,
                        "Y": y,
                        "Estado": status
                    })
                st.dataframe(point_data)
    
    # Download section
    st.subheader("Descargar Resultados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Save binary mask
        mask_pil = Image.fromarray(binary_mask * 255)
        mask_bytes = mask_pil.tobytes()
        st.download_button(
            label="Descargar máscara binaria",
            data=mask_bytes,
            file_name=f"mask_{uploaded_file.name.split('.')[0]}.png",
            mime="image/png",
            key="download_mask"
        )
    
    with col2:
        if show_heatmap:
            # Save heatmap
            heatmap_pil = Image.fromarray(heatmap)
            heatmap_bytes = heatmap_pil.tobytes()
            st.download_button(
                label="Descargar mapa de calor",
                data=heatmap_bytes,
                file_name=f"heatmap_{uploaded_file.name.split('.')[0]}.png",
                mime="image/png",
                key="download_heatmap"
            )

else:
    st.info("👈 Sube una imagen usando el panel lateral para comenzar")
    
    # Show example images if available
    example_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images_to_predict")
    if os.path.exists(example_dir):
        example_files = [f for f in os.listdir(example_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if example_files:
            st.subheader("Imágenes de ejemplo disponibles:")
            cols = st.columns(min(3, len(example_files)))
            for i, file in enumerate(example_files[:3]):  # Show first 3 files
                with cols[i]:
                    example_path = os.path.join(example_dir, file)
                    example_img = Image.open(example_path)
                    st.image(example_img, caption=file, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("🔬 **Desarrollado para análisis de plantas de maíz**")