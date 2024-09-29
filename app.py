import cv2
import numpy as np
import os
from flask import Flask, request, render_template, send_file

app = Flask(__name__)

# Function to check if the image is likely to be a solar panel
def is_solar_panel(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    panel_like_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Adjust threshold for minimum area of solar panel-like structures
            panel_like_contours.append(contour)

    return len(panel_like_contours) > 0

# Function to crop the image to a specific region (x, y, width, height)
def crop_image(image, x, y, width, height):
    return image[y:y + height, x:x + width]

# Function to detect dust regions and calculate dust percentage
def analyze_image(filepath):
    image = cv2.imread(filepath)

    if image is None:
        return 0, None

    # Check if the image is a solar panel
    if not is_solar_panel(image):
        return 'Not a valid image', None

    # Crop the image to focus on the central part (you can adjust the crop area as needed)
    height, width = image.shape[:2]
    x_start = int(width * 0.1)  # Starting x position (10% from left)
    y_start = int(height * 0.1)  # Starting y position (10% from top)
    crop_width = int(width * 0.8)  # Crop width (80% of original width)
    crop_height = int(height * 0.8)  # Crop height (80% of original height)
    cropped_image = crop_image(image, x_start, y_start, crop_width, crop_height)

    # Convert the cropped image to grayscale
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to detect darker (dusty) regions
    _, thresholded_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dusty_regions = cropped_image.copy()
    dust_pixels = 0
    total_pixels = cropped_image.shape[0] * cropped_image.shape[1]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Adjust this threshold to ignore smaller dust particles
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(dusty_regions, (x, y), (x + w, y + h), (0, 0, 255), 2)
            dust_pixels += area

    dust_percentage = (dust_pixels / total_pixels) * 100

    output_path = os.path.join('uploads', 'dusty_' + os.path.basename(filepath))
    cv2.imwrite(output_path, dusty_regions)

    return dust_percentage, output_path

# Route to handle image upload and analysis
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    filename = file.filename

    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    filepath = os.path.join('uploads', filename)
    file.save(filepath)

    dust_percentage, output_path = analyze_image(filepath)

    if dust_percentage == 'Not a valid image':
        return render_template('index.html', result='Not a valid image. Please upload a solar panel image.', image_url=None)

    if dust_percentage < 10:  # Use a lower threshold for clean panels
        result = 'Dust areas not found, panel is clean.'
        processed_image_url = None
    else:
        result = 'Dirty'
        processed_image_url = output_path

    return render_template(
        'index.html', 
        result=result, 
        cleanliness_percentage=100 - dust_percentage, 
        original_image_url=filepath,   
        processed_image_url=processed_image_url  
    )

@app.route('/uploads/<filename>')
def display_image(filename):
    return send_file(os.path.join('uploads', filename), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
