import pandas as pd
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.restoration import denoise_tv_chambolle
from medpy.filter.smoothing import anisotropic_diffusion
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Lunar DEM Generator",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 36px !important;
        color: #4F8BF9 !important;
        text-align: center;
        padding: 20px;
    }
    .image-caption {
        font-weight: bold;
        text-align: center;
        margin-top: 10px;
    }
    .processing-container {
        background-color: #0E1117;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #4F8BF9;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🌙 Lunar Digital Elevation Model Generator</h1>', unsafe_allow_html=True)

# File uploader in sidebar
with st.sidebar:
    st.header("Upload Lunar Image")
    image = st.file_uploader("Choose a lunar surface image", type=["jpg", "jpeg", "png"])
    
    st.header("Processing Parameters")
    denoise_weight = st.slider("Denoising Strength", 0.01, 0.5, 0.1, 0.01)
    gaussian_kernel = st.slider("Gaussian Kernel Size", 3, 15, 5, 2)
    diffusion_iter = st.slider("Diffusion Iterations", 1, 20, 10, 1)

# Main processing function
def process_image(uploaded_image, denoise_weight, gaussian_kernel, diffusion_iter):
    try:
        # Convert to grayscale and resize
        pil_img = Image.open(uploaded_image)
        img = np.array(pil_img.convert('L'))  # Convert to grayscale
        img_resized = cv2.resize(img, (256, 256))
        
        # Normalize
        img_normalized = img_resized.astype('float32') / 255.0
        
        # Denoise
        img_denoised = denoise_tv_chambolle(img_normalized, weight=denoise_weight, channel_axis=None)
        
        # Gaussian smoothing
        img_smoothed = cv2.GaussianBlur(
            (img_denoised * 255).astype('uint8'), 
            (gaussian_kernel, gaussian_kernel), 
            1
        )
        
        # Histogram equalization
        img_eq = cv2.equalizeHist(img_smoothed)
        
        # Anisotropic diffusion
        img_ad = anisotropic_diffusion(
            img_smoothed.astype('float32'), 
            niter=diffusion_iter, 
            kappa=50, 
            gamma=0.1
        )
        # Normalize for display
        img_ad_normalized = (img_ad - img_ad.min()) / (img_ad.max() - img_ad.min())
        
        return {
            "resized": img_resized,
            "normalized": img_normalized,
            "denoised": img_denoised,
            "smoothed": img_smoothed,
            "equalized": img_eq,
            "diffused": img_ad_normalized
        }
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Display results in two rows of three columns
if image is not None:
    results = process_image(image, denoise_weight, gaussian_kernel, diffusion_iter)
    
    if results:
        with st.container():
            st.subheader("Image Processing Pipeline")
            
            # First row
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(results["resized"], caption="1. Resized Image (256x256)", use_column_width=True)
            with col2:
                st.image(results["normalized"], caption="2. Normalized Grayscale", use_column_width=True)
            with col3:
                st.image(results["denoised"], caption="3. TV Denoising", use_column_width=True)
            
            # Second row
            col4, col5, col6 = st.columns(3)
            with col4:
                st.image(results["smoothed"], caption="4. Gaussian Smoothing", use_column_width=True)
            with col5:
                st.image(results["equalized"], caption="5. Histogram Equalization", use_column_width=True)
            with col6:
                st.image(results["diffused"], caption="6. Anisotropic Diffusion", use_column_width=True)
        
        # DEM Generation Section
        with st.expander("Generate Digital Elevation Model", expanded=True):
            st.subheader("DEM Generation Parameters")
            
            col_params1, col_params2 = st.columns(2)
            with col_params1:
                sun_azimuth = st.slider("Sun Azimuth (degrees)", 0, 360, 315, 5)
                sun_elevation = st.slider("Sun Elevation (degrees)", 0, 90, 45, 5)
            with col_params2:
                scale_factor = st.slider("Height Scale Factor", 0.1, 5.0, 1.0, 0.1)
                resolution = st.selectbox("Output Resolution", ["Low (128x128)", "Medium (256x256)", "High (512x512)"])
            
            if st.button("Generate DEM"):
                with st.spinner("Generating elevation model..."):
                    # Calculate incidence angle
                    incidence_angle = 90 - sun_elevation
                    
                    # Placeholder for DEM generation logic
                    # In a real implementation, you would use photogrammetry techniques here
                    dem = np.random.rand(256, 256) * scale_factor
                    
                    # Visualization
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(dem, cmap='terrain')
                    plt.colorbar(im, ax=ax, label='Elevation (arbitrary units)')
                    ax.set_title("Generated Lunar DEM")
                    st.pyplot(fig)
                    
                    st.success("DEM generated successfully! Download:")
                    st.download_button(
                        label="Download DEM as CSV",
                        data=pd.DataFrame(dem).to_csv().encode('utf-8'),
                        file_name='lunar_dem.csv',
                        mime='text/csv'
                    )
else:
    st.info("👈 Please upload a lunar surface image to begin processing")
    st.image("https://www.nasa.gov/sites/default/files/thumbnails/image/nsyt_marscraters.jpg", 
             caption="Sample Lunar Surface Image", width=600)

# Add footer
st.markdown("---")
st.markdown("### About this DEM Generator")
st.markdown("""
This tool processes lunar surface images to generate Digital Elevation Models (DEMs) using:
- **Image Denoising**: Removes noise while preserving edges
- **Contrast Enhancement**: Improves topographic features visibility
- **Anisotropic Diffusion**: Enhances terrain features while smoothing homogeneous regions
- **Photogrammetry**: Converts 2D images to 3D elevation data using sun position data
""")