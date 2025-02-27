import os
import time
from flask import Flask, render_template, request

from werkzeug.utils import secure_filename, redirect

from inspection.segmentation import SegmentationExplainer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['LABEL_FOLDER'] = 'static/labels/'
app.config['SEG_RES_FOLDER'] = 'static/results/'
app.config['CLASS_RES_FOLDER'] = 'static/results/classification/'


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        # Check if a label file is uploaded
        label_file = request.files['label_file'] if request.files.get('label_file').filename != '' else None
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if label_file is not None:
                label_filename = secure_filename(label_file.filename)
                label_filepath = os.path.join(app.config['LABEL_FOLDER'], label_filename)
                label_file.save(label_filepath)
            else:
                label_filepath = None

            # Get the chosen options
            problem = request.form.get("problem")
            model = request.form.get("model")
            xai = request.form.get("explainable_ai")
            category = request.form.get("category")

            if problem == 'classification':
                result_filename = f'result_{xai}_{model}_{category}_{filename}'
                result_filepath = os.path.join(app.config['CLASS_RES_FOLDER'], result_filename)

                if not os.path.isfile(result_filepath):
                    explanation = ClassificationExplainer().explain(image_path=filepath)


            elif problem == 'segmentation':
                result_filename = f'result_{xai}_{model}_{category}_{filename}'
                result_filepath = os.path.join(app.config['SEG_RES_FOLDER'], result_filename)

                segment_filename = f'segment_{category}_{model}_{filename}'
                segment_filepath = os.path.join(app.config['SEG_RES_FOLDER'], segment_filename)

                coco_filename = f'coco_{category}_{model}_{filename}'
                coco_filepath = os.path.join(app.config['SEG_RES_FOLDER'], coco_filename)

                start_time = time.time()
                # Process the image with the SegmentationExplainer class
                # Check if result_filepath is available or not
                if not os.path.isfile(result_filepath) or os.path.isfile(coco_filepath):
                    segmentation_image, cam_image, coco_image = \
                        SegmentationExplainer(model=model).explain(image_path=filepath,
                                                                   category=category,
                                                                   label_path=label_filepath,
                                                                   xai=xai)
                    # Save the cam_image result to a file
                    cam_image.save(result_filepath)
                    segmentation_image.save(segment_filepath)
                    if coco_image is not None:
                        coco_image.save(coco_filepath)
                else:
                    coco_image = None
                processing_time = time.time() - start_time
                print(f'Processing time: {processing_time}')

                if coco_image is None:
                    return render_template('result.html', image=filepath, segmentation=segment_filepath,
                                           result=result_filepath)
                else:
                    return render_template('result.html', image=filepath, segmentation=segment_filepath,
                                           result=result_filepath,
                                           coco_image=coco_filepath)

    return render_template('index.html')


@app.route('/xai_eval', methods=['GET', 'POST'])
def xai_eval():
    return


@app.route('/improvement', methods=['GET', 'POST'])
def improvement():
    return render_template('improvement.html')


if __name__ == '__main__':
    app.run(debug=True)
