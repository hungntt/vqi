import os

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename, redirect

from inspection.gradcam import GradCamSegmentation

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ANNOTATION_FOLDER'] = 'static/annotations/'
app.config['RESULT_FOLDER'] = 'static/results/'


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['image-file']
        label = request.files['label-file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            labelname = secure_filename(label.filename)
            label_path = os.path.join(app.config['ANNOTATION_FOLDER'], labelname)
            label.save(label_path)

            # Get the chosen options
            xai = request.form.get("explainable_ai")

            # Process the image with the GradCamSegmentation class
            segmentation_image, cam_image, coco_image = GradCamSegmentation().process_image(image_path=image_path,
                                                                                            label_path=label_path,
                                                                                            is_url=False,
                                                                                            xai=xai, )
            # Save the cam_image result to a file
            result_filename = 'result_' + filename
            result_filepath = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            cam_image.save(result_filepath)

            segment_filename = 'segment_' + filename
            segment_filepath = os.path.join(app.config['RESULT_FOLDER'], segment_filename)
            segmentation_image.save(segment_filepath)

            return render_template('result.html',
                                   image=image_path,
                                   segmentation=segment_filepath,
                                   result=result_filepath,
                                   coco_image=coco_image)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
