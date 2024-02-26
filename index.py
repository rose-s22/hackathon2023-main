from flask import render_template, Flask, request
from werkzeug.utils import secure_filename
import os
import pixelize

app = Flask(__name__)
upload_folder = os.path.join('static', 'uploads')
app.config['UPLOAD'] = upload_folder

if (not os.path.exists(upload_folder)):
    os.mkdir(upload_folder)

@app.route('/')
def render():
    return render_template('index.html')

@app.route('/generate', methods = ['POST'])
def file_uploader():
    if request.method == "POST":
        file = request.files["fileToUpload"]
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img = os.path.join(app.config['UPLOAD'], filename)

        pixelizer = pixelize.pixelator(img, 1000)
        pixelated_img = pixelizer.get_pixelated()

        return render_template('index.html', image=pixelated_img[0], coloredImage = pixelated_img[1])

if __name__ == '__main__':
   app.run(debug = False, host='0.0.0.0')
