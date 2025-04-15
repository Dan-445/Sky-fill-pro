import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Create the output directory if it doesn't exist
output_dir = "sky_filled"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_image(image_file):
    try:
        print(f"\nProcessing: {image_file}", flush=True)
        
        # Read the image
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image at {image_path}")
            return None

        height, width = image.shape[:2]
        print(f"Image dimensions: {width}x{height}", flush=True)
        
        # STEP 1: Ultra-precise mask detection for ONLY the missing spaces
        print("Precise mask detection...", flush=True)
        
        # Convert to multiple color spaces for better analysis
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create very precise mask for true black pixels (missing areas only)
        # Use a tighter threshold to avoid affecting non-missing areas
        black_mask = cv2.inRange(hsv_image[:,:,2], 0, 10)  # Only truly dark regions
        
        # For vertical white strips in sky, create a separate mask
        # Focus only on top portion where sky is located
        top_region = image[:height//3, :, :]
        hsv_top = hsv_image[:height//3, :, :]
        
        # Detect bright vertical strips (white lines)
        # Look for high value (brightness) with low saturation
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
        print("Analyzing surrounding colors...", flush=True)
        
        # Create a color reference image for sampling nearby colors
        color_reference = image.copy()
        
        # Prepare for inpainting with contextual information
        kernel_small = np.ones((3,3), np.uint8)
        dilated_mask = cv2.dilate(full_mask, kernel_small, iterations=2)
        
        # STEP 3: Ultra-precise content-aware inpainting - ONLY on missing areas
        print("Precise inpainting of missing areas only...", flush=True)
        
        # Create a copy of the original image for filling
        filled_image = image.copy()
        
        # First pass: color initialization for better results
        # Get the average color of surrounding pixels for each missing area
        expanded_mask = cv2.dilate(dilated_mask, np.ones((9,9), np.uint8))
        boundary = cv2.subtract(expanded_mask, dilated_mask)
        
        # Initialize with average color based on position
        # We'll create a localized color map
        print("Creating localized color map...", flush=True)
        
        # Divide the image into a grid for localized color matching
        grid_size = 30  # pixels
        for y in range(0, height, grid_size):
            for x in range(0, width, grid_size):
                # Get local region
                y_end = min(y + grid_size, height)
                x_end = min(x + grid_size, width)
                
                # Get local masks
                local_mask = dilated_mask[y:y_end, x:x_end]
                local_boundary = boundary[y:y_end, x:x_end]
                
                # If there are pixels to fill in this region
                if np.sum(local_mask) > 0 and np.sum(local_boundary) > 0:
                    # Get local reference colors from boundary
                    local_region = image[y:y_end, x:x_end]
                    local_avg_color = [0, 0, 0]
                    
                    for c in range(3):
                        valid_colors = local_region[:,:,c][local_boundary > 0]
                        if len(valid_colors) > 0:
                            # Use median for better color robustness
                            local_avg_color[c] = int(np.median(valid_colors))
                    
                    # Initialize missing pixels with local color
                    local_fill = filled_image[y:y_end, x:x_end].copy()
                    for c in range(3):
                        local_fill[:,:,c] = np.where(local_mask > 0, local_avg_color[c], local_fill[:,:,c])
                    
                    filled_image[y:y_end, x:x_end] = local_fill
        
        # STEP 4: Apply multiple progressive inpainting passes ONLY to the masked areas
        print("Progressive inpainting passes...", flush=True)
        
        # First pass with small radius - detail preservation
        filled_image = cv2.inpaint(filled_image, dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        # Second pass with medium radius - structure consistency
        filled_image = cv2.inpaint(filled_image, dilated_mask, inpaintRadius=7, flags=cv2.INPAINT_NS)
        
        # Third pass focusing on smoothness
        filled_image = cv2.inpaint(filled_image, dilated_mask, inpaintRadius=11, flags=cv2.INPAINT_TELEA)
        
        # STEP 5: Ultra-smooth blending ONLY at the boundaries between original and filled
        print("Ultra-smooth boundary blending...", flush=True)
        
        # Create a narrow transition zone just at the boundaries
        transition_zone = cv2.dilate(dilated_mask, np.ones((5,5), np.uint8), iterations=1)
        transition_zone = cv2.subtract(transition_zone, dilated_mask)
        
        # Apply a very targeted bilateral filter only at the transition zone
        smoothed = cv2.bilateralFilter(filled_image, d=9, sigmaColor=50, sigmaSpace=50)
        
        # Create a smooth gradient for the transition (feathered mask)
        transition_float = transition_zone.astype(float) / 255
        transition_float = cv2.GaussianBlur(transition_float, (9, 9), 0)
        transition_float = np.dstack([transition_float] * 3)
        
        # Blend only at the transition zone
        filled_image = filled_image * (1 - transition_float) + smoothed * transition_float
        
        # STEP 6: Special anti-white artifact processing
        print("Removing white artifacts...", flush=True)
        
        # Convert to LAB to better handle luminance separately from color
        filled_lab = cv2.cvtColor(filled_image.astype(np.uint8), cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(filled_lab)
        
        # Identify overly bright areas within the filled region
        # Get statistics of surrounding non-filled areas for reference
        surrounding_l = l[boundary > 0]
        if len(surrounding_l) > 0:
            mean_l = np.mean(surrounding_l)
            std_l = np.std(surrounding_l)
            
            # If a filled pixel is too bright compared to surroundings, tone it down
            too_bright = np.logical_and(l > mean_l + 1.5*std_l, dilated_mask > 0)
            l[too_bright] = np.clip(l[too_bright], 0, mean_l + std_l)
            
        # Merge channels and convert back
        filled_lab = cv2.merge((l, a, b))
        filled_image = cv2.cvtColor(filled_lab, cv2.COLOR_LAB2BGR)
        
        # STEP 7: Final check to ensure ALL missing spaces are filled
        print("Final verification of complete filling...", flush=True)
        
        # Check if any black pixels remain in the filled image
        hsv_result = cv2.cvtColor(filled_image.astype(np.uint8), cv2.COLOR_BGR2HSV)
        remaining_black = cv2.inRange(hsv_result[:,:,2], 0, 5)
        
        # If any remain, do one final targeted pass
        if np.sum(remaining_black) > 0:
            print("Filling remaining black spots...", flush=True)
            # Dilate to ensure coverage
            final_mask = cv2.dilate(remaining_black, kernel_small, iterations=2)
            # Final pass
            filled_image = cv2.inpaint(filled_image, final_mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
        
        # Final cleanup to ensure type consistency
        filled_image = np.clip(filled_image, 0, 255).astype(np.uint8)
        
        # Save the result
        print("Saving result...", flush=True)
        base_name = os.path.splitext(image_file)[0]
        output_path = os.path.join(output_dir, f'filled_{base_name}.png')
        cv2.imwrite(output_path, filled_image)
        
        return image, full_mask, filled_image
        
    except Exception as e:
        print(f"Error processing {image_file}: {str(e)}", flush=True)
        return None
        
# Directory containing your images
image_dir = "NoSky"

# Get list of all images in the directory
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

print(f"Found {len(image_files)} images to process", flush=True)

# Process images one by one
for idx, image_file in enumerate(image_files, 1):
    print(f"\nProcessing image {idx}/{len(image_files)}", flush=True)
    
    result = process_image(image_file)
    
    if result is not None:
        image, mask, filled_image = result
        
        # Convert images for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        filled_image_rgb = cv2.cvtColor(filled_image, cv2.COLOR_BGR2RGB)

        # Visualization
        plt.figure(figsize=(15, 5))
        
        # Original image with black areas highlighted
        plt.subplot(1, 3, 1)
        plt.imshow(image_rgb)
        colored_mask = np.zeros((*image_rgb.shape[:2], 4))
        colored_mask[mask > 0] = [1, 0.7, 0.7, 0.5]
        plt.imshow(colored_mask)
        plt.title('Black Areas Highlighted')
        plt.axis('off')

        # Original image
        plt.subplot(1, 3, 2)
        plt.imshow(image_rgb)
        plt.title('Original Image')
        plt.axis('off')

        # Filled image
        plt.subplot(1, 3, 3)
        plt.imshow(filled_image_rgb)
        plt.title('Filled Image')
        plt.axis('off')

        # Remove this line to not save the comparison visualization
        # comparison_path = os.path.join(output_dir, f'comparison_{image_file.split(".")[0]}.png')
        # plt.savefig(comparison_path, bbox_inches='tight', dpi=300)
        
        plt.show()
        plt.close()
        
        print(f"Completed processing image {idx}/{len(image_files)}", flush=True)
        input("Press Enter for next image (Ctrl+C to stop)...")
        

print("\nAll images processed successfully!", flush=True)

# Add this after the main processing loop
def add_smoothing_brush_tool():
    print("\nInteractive Brush Smoothing Tool")
    print("--------------------------------")
    print("Use this tool to further refine filled areas.")
    print("CONTROLS:")
    print("  Left Mouse Button: Apply smoothing brush")
    print("  Mouse Wheel: Change brush size")
    print("  + / -: Increase/decrease smoothing strength")
    print("  s: Save current result")
    print("  r: Reset to original filled image")
    print("  q: Quit brush mode")
    
    # Select the last processed image or let user specify
    while True:
        choice = input("Enter image number to refine (or 'q' to quit): ")
        if choice.lower() == 'q':
            return
        
        try:
            idx = int(choice)
            if 1 <= idx <= len(image_files):
                image_file = image_files[idx-1]
                break
            else:
                print(f"Please enter a number between 1 and {len(image_files)}")
        except ValueError:
            print("Please enter a valid number")
    
    # Load the filled image
    base_name = os.path.splitext(image_file)[0]
    filled_path = os.path.join(output_dir, f'filled_{base_name}.png')
    
    if not os.path.exists(filled_path):
        print(f"Filled image not found: {filled_path}")
        return
        
    # Load the original image (for mask creation and reference)
    original_path = os.path.join(image_dir, image_file)
    original = cv2.imread(original_path)
    filled_image = cv2.imread(filled_path)
    
    if original is None or filled_image is None:
        print("Error loading images")
        return
    
    # Create a mask of the filled areas by comparing original and filled
    # Convert to grayscale for comparison
    gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_filled = cv2.cvtColor(filled_image, cv2.COLOR_BGR2GRAY)
    
    # Find difference between original and filled
    diff = cv2.absdiff(gray_original, gray_filled)
    _, fill_mask = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)
    
    # Dilate mask slightly to ensure full coverage
    kernel = np.ones((5,5), np.uint8)
    fill_mask = cv2.dilate(fill_mask, kernel, iterations=2)
    
    # Create a copy for resetting
    original_filled = filled_image.copy()
    current_image = filled_image.copy()
    
    # Brush parameters
    brush_size = 15
    smoothing_strength = 5
    drawing = False
    
    # Function for mouse callback
    def brush_callback(event, x, y, flags, param):
        nonlocal drawing, current_image, brush_size
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            apply_smoothing(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            apply_smoothing(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            
    # Function to apply smoothing only to filled areas
    def apply_smoothing(x, y):
        nonlocal current_image, fill_mask, brush_size, smoothing_strength
        
        # Create a circular brush mask
        brush_mask = np.zeros(current_image.shape[:2], dtype=np.uint8)
        cv2.circle(brush_mask, (x, y), brush_size, 255, -1)
        
        # Only affect filled areas (intersection of brush and fill_mask)
        brush_mask = cv2.bitwise_and(brush_mask, fill_mask)
        
        # Skip if no overlap with filled areas
        if np.sum(brush_mask) == 0:
            return
            
        # Apply bilateral filter with brush_mask as ROI
        roi = current_image.copy()
        
        # For stronger smoothing, do multiple passes
        for _ in range(smoothing_strength):
            # Apply edge-preserving filter to the entire image
            smoothed = cv2.bilateralFilter(roi, d=9, sigmaColor=75, sigmaSpace=75)
            
            # Create a float version of the brush mask for blending
            brush_float = brush_mask.astype(float) / 255
            brush_float = cv2.GaussianBlur(brush_float, (21, 21), 0)
            brush_float = np.dstack([brush_float] * 3)
            
            # Blend original and smoothed based on brush
            roi = roi * (1 - brush_float) + smoothed * brush_float
            
        # Update current image
        current_image = roi.astype(np.uint8)
    
    # Create window and set mouse callback
    cv2.namedWindow('Smoothing Brush Tool')
    cv2.setMouseCallback('Smoothing Brush Tool', brush_callback)
    
    # Main loop for brush tool
    while True:
        # Create a display image with brush preview
        display = current_image.copy()
        
        # Show brush cursor
        mouse_x, mouse_y = -1, -1
        mouse_pos = cv2.getWindowProperty('Smoothing Brush Tool', cv2.WND_PROP_ASPECT_RATIO)
        if mouse_pos >= 0:  # Only if window is in focus
            # Get current mouse position - this is a bit hacky in OpenCV
            # Better implementation would use a global variable updated by the callback
            pass
        
        # Draw brush outline
        h, w = display.shape[:2]
        info_text = f"Brush Size: {brush_size} | Strength: {smoothing_strength}"
        cv2.putText(display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Show image
        cv2.imshow('Smoothing Brush Tool', display)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
                break
        elif key == ord('+') or key == ord('='):
            smoothing_strength = min(smoothing_strength + 1, 10)
        elif key == ord('-'):
            smoothing_strength = max(smoothing_strength - 1, 1)
        elif key == ord('s'):
            # Save current result
            output_path = os.path.join(output_dir, f'refined_{base_name}.png')
            cv2.imwrite(output_path, current_image)
            print(f"Saved refined image to: {output_path}")
        elif key == ord('r'):
            # Reset to original filled image
            current_image = original_filled.copy()
        elif key == ord('['):
            brush_size = max(brush_size - 5, 5)
        elif key == ord(']'):
            brush_size = min(brush_size + 5, 100)
    
    cv2.destroyAllWindows()
    
    # Ask if user wants to save the final result
    choice = input("Save the refined image? (y/n): ")
    if choice.lower() == 'y':
        output_path = os.path.join(output_dir, f'refined_{base_name}.png')
        cv2.imwrite(output_path, current_image)
        print(f"Saved refined image to: {output_path}")
    
    print("Brush tool closed")

# Add this at the end of your main script
print("\nAll images processed successfully!", flush=True)
print("Would you like to use the smoothing brush tool to refine filled areas? (y/n)")
choice = input()
if choice.lower() == 'y':
    add_smoothing_brush_tool()