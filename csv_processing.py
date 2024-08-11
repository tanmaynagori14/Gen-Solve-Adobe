from flask import Flask, request, redirect, url_for, send_from_directory
import numpy as np
import svgwrite
from svgpathtools import svg2paths
from PIL import Image, ImageDraw
import cv2
import os

app = Flask(__name__)

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def convert_to_svg(path_XYs, svg_path='static/output.svg'):
    dwg = svgwrite.Drawing(svg_path, profile='tiny')
    for i, XYs in enumerate(path_XYs):
        for XY in XYs:
            path = 'M' + ' '.join([f"{x},{y}" for x, y in XY]) + ' Z'
            dwg.add(dwg.path(d=path, fill='none', stroke='black', stroke_width=2))
    dwg.save()

def svg_to_image(svg_path, image_path='static/output.png'):
    img = Image.new('RGB', (800, 800), 'white')
    draw = ImageDraw.Draw(img)

    paths, attributes = svg2paths(svg_path)
    for path in paths:
        for segment in path:
            if segment.__class__.__name__ == 'Line':
                draw.line((segment.start.real, segment.start.imag, segment.end.real, segment.end.imag), fill='black', width=2)
            elif segment.__class__.__name__ == 'CubicBezier':
                draw.line((segment.start.real, segment.start.imag, segment.end.real, segment.end.imag), fill='black', width=2)

    img.save(image_path)
    return img

def detect_shapes_from_image(image_path, output_image_path='static/shapes_detected.png'):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    height, width = image.shape

    blank_image = np.ones((height, width, 3), dtype=np.uint8) * 255

    _, thresh_image = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    drawn_contours = []

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        approx_tuple = tuple(map(tuple, approx.reshape(-1, 2)))

        if approx_tuple not in drawn_contours:
            drawn_contours.append(approx_tuple)

            if len(approx) == 3:
                cv2.drawContours(blank_image, [approx], -1, (0, 0, 0), 2)
            elif len(approx) == 4:
                cv2.drawContours(blank_image, [approx], -1, (0, 0, 0), 2)
            elif len(approx) == 5:
                cv2.drawContours(blank_image, [approx], -1, (0, 0, 0), 2)
            elif len(approx) == 6:
                cv2.drawContours(blank_image, [approx], -1, (0, 0, 0), 2)
            elif len(approx) == 10:
                cv2.drawContours(blank_image, [approx], -1, (0, 0, 0), 2)
            else:
                circles = cv2.HoughCircles(thresh_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=200)
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    if len(circles) > 0:
                        x, y, r = circles[0]
                        cv2.circle(blank_image, (x, y), r, (0, 0, 0), 2)
                    continue

    blank_image = cv2.flip(blank_image, 0)

    # Save the result
    cv2.imwrite(output_image_path, blank_image)

def image_to_csv(image_path, csv_path='static/output_detected.csv'):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh_image = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    height, width = image.shape

    with open(csv_path, 'w') as file:
        for path_id, contour in enumerate(contours):
            for point in contour[:, 0, :]:
                flipped_y = height - point[1]
                file.write(f"{path_id},{point[0]},{flipped_y}\n")

def csv_to_svg_to_csv(input_csv_path, output_svg_path='static/output.svg', output_image_path='static/output.png', output_csv_path='static/output_detected.csv'):
    path_XYs = read_csv(input_csv_path)
    convert_to_svg(path_XYs, output_svg_path)
    svg_to_image(output_svg_path, output_image_path)
    detect_shapes_from_image(output_image_path)
    image_to_csv(output_image_path, output_csv_path)

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Process the file
    csv_to_svg_to_csv(file_path)

    return redirect(url_for('static', filename='output_detected.csv'))

@app.route('/static/<path:filename>')
def download_file(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    app.run(debug=True)
