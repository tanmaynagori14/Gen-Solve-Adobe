from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os
from mirror_symmetry import detect_mirror_line
from csv_processing import csv_to_svg_to_csv

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'svg', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Process image
        detect_mirror_line(file_path)
        return redirect(url_for('result', filename='Detected Mirror Line.png', type='symmetry'))
    return redirect(request.url)

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Process CSV
        csv_to_svg_to_csv(file_path)
        return redirect(url_for('result', filename='shapes_detected.png', type='regularized'))
    return redirect(request.url)


# @app.route('/result/<filename>')
@app.route('/result')
def result():
    filename = request.args.get('filename')
    result_type = request.args.get('type')  # 'symmetry' or 'regularized'
    
    if result_type == 'symmetry':
        result_filename = 'Detected_Mirror_Line.png'
    elif result_type == 'regularized':
        result_filename = 'shapes_detected.png'
    else:
        result_filename = None
    
    return render_template('result.html', filename=result_filename, type=result_type)



if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
