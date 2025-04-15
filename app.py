import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

def convert_to_base64(image, max_dimension=2048):
    """Convert image to base64 with optimized size"""
    # Convert from BGR to RGB
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize if too large
    height, width = image.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    
    # Save to BytesIO object with optimized quality
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG", quality=85, optimize=True)
    
    # Get base64 string
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def create_360_viewer(image):
    """Create an optimized 360-degree viewer"""
    # Convert image to base64 with optimized size
    base64_image = convert_to_base64(image)
    
    # HTML and JavaScript for Pannellum viewer with optimized settings
    html_content = f"""
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/pannellum@2.5.6/build/pannellum.css"/>
        <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/pannellum@2.5.6/build/pannellum.js"></script>
        <style>
            #panorama {{
                width: 100%;
                height: 500px;
                border-radius: 10px;
            }}
            .pnlm-loading {{
                display: none;
            }}
        </style>
        <div id="panorama"></div>
        <script>
            pannellum.viewer('panorama', {{
                "type": "equirectangular",
                "panorama": "data:image/jpeg;base64,{base64_image}",
                "autoLoad": true,
                "autoRotate": -2,
                "compass": true,
                "showZoomCtrl": true,
                "showFullscreenCtrl": true,
                "mouseZoom": true,
                "friction": 0.2,
                "hfov": 100,
                "minHfov": 50,
                "maxHfov": 120,
                "pitch": 0,
                "minPitch": -90,
                "maxPitch": 90,
                "yaw": 0,
                "minYaw": -180,
                "maxYaw": 180,
                "draggable": true,
                "disableKeyboardCtrl": false,
                "keyboardZoom": true,
                "mouseZoom": true,
                "showControls": true,
                "verticalDrag": true,
                "preview": null,
                "dynamic": true,
                "dynamicQuality": 75,
                "dynamicMinQuality": 30,
                "dynamicMinHfov": 50,
                "dynamicMaxHfov": 120,
                "dynamicMinPitch": -90,
                "dynamicMaxPitch": 90,
                "dynamicMinYaw": -180,
                "dynamicMaxYaw": 180,
                "dynamicMinZoom": 0.5,
                "dynamicMaxZoom": 2.0,
                "dynamicMinFps": 20,
                "dynamicMaxFps": 30
            }});
        </script>
    """
    return html_content

def process_single_image(image):
    """Process image to fill sky areas using advanced techniques"""
    try:
        height, width = image.shape[:2]
        
        # STEP 1: Ultra-precise mask detection for ONLY the missing spaces
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create very precise mask for true black pixels (missing areas only)
        black_mask = cv2.inRange(hsv_image[:,:,2], 0, 10)  # Only truly dark regions
        
        # For vertical white strips in sky, create a separate mask
        top_region = image[:height//3, :, :]
        hsv_top = hsv_image[:height//3, :, :]
        
        # Detect bright vertical strips (white lines)
        white_mask_top = cv2.inRange(hsv_top, 
                               np.array([0, 0, 240]),  # Low saturation, high value
                               np.array([180, 30, 255]))  # Any hue, low saturation, high value
        
        # Use morphology to isolate vertical structures
        vertical_kernel = np.ones((15, 1), np.uint8)
        vertical_white = cv2.morphologyEx(white_mask_top, cv2.MORPH_CLOSE, vertical_kernel)
        vertical_white = cv2.dilate(vertical_white, np.ones((5, 1), np.uint8), iterations=1)
        
        # Create full mask combining black areas and white vertical strips
        full_mask = np.zeros((height, width), dtype=np.uint8)
        full_mask[:height//3, :] = vertical_white
        full_mask = cv2.bitwise_or(full_mask, black_mask)
        
        # STEP 2: Neighborhood color analysis for perfect color matching
        color_reference = image.copy()
        kernel_small = np.ones((3,3), np.uint8)
        dilated_mask = cv2.dilate(full_mask, kernel_small, iterations=2)
        
        # STEP 3: Ultra-precise content-aware inpainting
        filled_image = image.copy()
        
        # First pass: color initialization for better results
        expanded_mask = cv2.dilate(dilated_mask, np.ones((9,9), np.uint8))
        boundary = cv2.subtract(expanded_mask, dilated_mask)
        
        # Initialize with average color based on position
        grid_size = 30  # pixels
        for y in range(0, height, grid_size):
            for x in range(0, width, grid_size):
                y_end = min(y + grid_size, height)
                x_end = min(x + grid_size, width)
                
                local_mask = dilated_mask[y:y_end, x:x_end]
                local_boundary = boundary[y:y_end, x:x_end]
                
                if np.sum(local_mask) > 0 and np.sum(local_boundary) > 0:
                    local_region = image[y:y_end, x:x_end]
                    local_avg_color = [0, 0, 0]
                    
                    for c in range(3):
                        valid_colors = local_region[:,:,c][local_boundary > 0]
                        if len(valid_colors) > 0:
                            local_avg_color[c] = int(np.median(valid_colors))
                    
                    local_fill = filled_image[y:y_end, x:x_end].copy()
                    for c in range(3):
                        local_fill[:,:,c] = np.where(local_mask > 0, local_avg_color[c], local_fill[:,:,c])
                    
                    filled_image[y:y_end, x:x_end] = local_fill
        
        # STEP 4: Apply multiple progressive inpainting passes
        filled_image = cv2.inpaint(filled_image, dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        filled_image = cv2.inpaint(filled_image, dilated_mask, inpaintRadius=7, flags=cv2.INPAINT_NS)
        filled_image = cv2.inpaint(filled_image, dilated_mask, inpaintRadius=11, flags=cv2.INPAINT_TELEA)
        
        # STEP 5: Ultra-smooth blending at boundaries
        transition_zone = cv2.dilate(dilated_mask, np.ones((5,5), np.uint8), iterations=1)
        transition_zone = cv2.subtract(transition_zone, dilated_mask)
        
        smoothed = cv2.bilateralFilter(filled_image, d=9, sigmaColor=50, sigmaSpace=50)
        transition_float = transition_zone.astype(float) / 255
        transition_float = cv2.GaussianBlur(transition_float, (9, 9), 0)
        transition_float = np.dstack([transition_float] * 3)
        
        filled_image = filled_image * (1 - transition_float) + smoothed * transition_float
        
        # STEP 6: Anti-white artifact processing
        filled_lab = cv2.cvtColor(filled_image.astype(np.uint8), cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(filled_lab)
        
        surrounding_l = l[boundary > 0]
        if len(surrounding_l) > 0:
            mean_l = np.mean(surrounding_l)
            std_l = np.std(surrounding_l)
            too_bright = np.logical_and(l > mean_l + 1.5*std_l, dilated_mask > 0)
            l[too_bright] = np.clip(l[too_bright], 0, mean_l + std_l)
            
        filled_lab = cv2.merge((l, a, b))
        filled_image = cv2.cvtColor(filled_lab, cv2.COLOR_LAB2BGR)
        
        # Final cleanup
        filled_image = np.clip(filled_image, 0, 255).astype(np.uint8)
        
        return filled_image, full_mask
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

def main():
    st.markdown("<h1 class='main-header'>🌤️ Sky Fill Pro 360°</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Fill Missing Sky Areas with Interactive 360° View</h3>", 
                unsafe_allow_html=True)

    # Add custom CSS
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            color: #1E88E5;
        }
        .stButton>button {
            width: 100%;
            background-color: #1E88E5;
            color: white;
        }
        .upload-prompt {
            text-align: center;
            padding: 20px;
            border: 2px dashed #1E88E5;
            border-radius: 10px;
            margin: 20px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a panoramic image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        try:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                st.error("Failed to load image. Please try another file.")
                return

            # Check if image is suitable for 360 viewing
            height, width = image.shape[:2]
            if width / height < 1.5:
                st.warning("This image might not be suitable for 360° viewing. Best results are achieved with panoramic images (2:1 aspect ratio).")

            # Create tabs
            tab1, tab2 = st.tabs(["Standard View", "Interactive 360° View"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Original Image")
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if st.button("Fill Sky Areas"):
                    with st.spinner('Processing...'):
                        filled_image, mask = process_single_image(image)
                        
                        if filled_image is not None:
                            with col2:
                                st.markdown("### Filled Image")
                                st.image(cv2.cvtColor(filled_image, cv2.COLOR_BGR2RGB))
                                
                                # Save button
                                filled_bytes = cv2.imencode('.png', filled_image)[1].tobytes()
                                st.download_button(
                                    label="Download Result",
                                    data=filled_bytes,
                                    file_name="filled_panorama.png",
                                    mime="image/png"
                                )
                            
                            # Store for 360 view
                            st.session_state.filled_image = filled_image
                            st.success("Processing complete! Check the Interactive 360° View tab.")
            
            with tab2:
                try:
                    st.markdown("### Original 360° View")
                    st.markdown("*Move your mouse to look around, scroll to zoom*")
                    html_content = create_360_viewer(image)
                    components.html(html_content, height=550)
                    
                    if 'filled_image' in st.session_state:
                        st.markdown("### Filled 360° View")
                        html_content_filled = create_360_viewer(st.session_state.filled_image)
                        components.html(html_content_filled, height=550)
                except Exception as e:
                    st.error(f"Error in 360° viewer: {str(e)}")
                    st.info("The standard view is still available above.")
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Instructions
    with st.expander("How to Use"):
        st.markdown("""
        1. Upload a panoramic image (preferably 360° equirectangular)
        2. Process the image to fill missing areas
        3. View both standard and interactive 360° versions
        4. In the 360° view:
           - Click and drag to look around
           - Scroll to zoom in/out
           - Click the fullscreen button for immersive viewing
        5. Download the processed result
        
        **For best 360° viewing results:**
        - Use equirectangular panoramic images
        - Ideal aspect ratio is 2:1 (width:height)
        - Images will automatically adjust for optimal viewing
        """)

if __name__ == "__main__":
    main()
