# Sky Fill Pro 360°

A powerful application for processing panoramic images with missing sky areas, featuring an interactive 360° viewer. This tool allows you to fill missing sky areas in panoramic images and view them in an immersive 360° environment.

## Features

- Upload and process panoramic images
- Fill missing sky areas with content-aware algorithms
- Interactive 360° viewing experience
- Real-time image processing
- Download processed images
- User-friendly interface
- Optimized performance

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Dan-445/sky-fill-pro.git
cd sky-fill-pro-360
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Packages

```bash
pip install -r requirements.txt
```

The following packages will be installed:
- streamlit
- opencv-python
- numpy
- pillow
- plotly

## Usage

### Running the Application

1. Activate your virtual environment (if you created one)
2. Run the application:

```bash
streamlit run app.py
```

3. The application will open in your default web browser at `http://localhost:8501`

### How to Use the Application

1. **Upload an Image**
   - Click on the file uploader
   - Select a panoramic image (preferably in 2:1 aspect ratio)
   - Supported formats: JPG, JPEG, PNG

2. **Process the Image**
   - Click the "Fill Sky Areas" button
   - Wait for the processing to complete
   - View the results in both standard and 360° views

3. **360° View Controls**
   - Click and drag to look around
   - Scroll to zoom in/out
   - Use arrow keys for navigation
   - Press Space to reset the view
   - Click the fullscreen button for immersive viewing

4. **Download Results**
   - Click the "Download Result" button to save the processed image

## Best Practices

1. **Image Requirements**
   - Use panoramic images with 2:1 aspect ratio
   - Recommended resolution: 4096x2048 pixels
   - Maximum file size: 20MB
   - Supported formats: JPG, JPEG, PNG

2. **For Best Results**
   - Ensure good lighting in the original image
   - Avoid extreme contrast in sky areas
   - Use high-quality source images
   - Check the aspect ratio before uploading

## Troubleshooting

1. **Application Won't Start**
   - Ensure all dependencies are installed
   - Check Python version (3.8 or higher)
   - Verify virtual environment activation

2. **Image Processing Issues**
   - Check image format and size
   - Ensure proper aspect ratio
   - Try reducing image resolution

3. **360° Viewer Problems**
   - Clear browser cache
   - Try a different browser
   - Check internet connection (for CDN resources)

## Performance Optimization

The application includes several optimizations:
- Image size reduction for faster loading
- Progressive quality loading
- Optimized memory usage
- Efficient processing algorithms

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## Acknowledgments

- Pannellum for the 360° viewer
- OpenCV for image processing
- Streamlit for the web interface 
