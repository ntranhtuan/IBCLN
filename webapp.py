import os
from flask import Flask, render_template, request
import torch
from PIL import Image
import torchvision.transforms as transforms

from options.test_options import TestOptions
from models import create_model
from util import util

# configure folders
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# initialize flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# ensure required options have default values when invoked without CLI
import sys
if '--dataroot' not in sys.argv:
    sys.argv.extend(['--dataroot', '.'])
if '--model' not in sys.argv:
    sys.argv.extend(['--model', 'IBCLN'])
if '--name' not in sys.argv:
    sys.argv.extend(['--name', 'IBCLN'])
if '--dataset_mode' not in sys.argv:
    sys.argv.extend(['--dataset_mode', 'resize_natural_3'])
if '--gpu_ids' not in sys.argv:
    sys.argv.extend(['--gpu_ids', '-1'])

# load the trained IBCLN model once when the server starts
opt = TestOptions().parse()
# if gpu_ids contains -1 we want CPU-only
if isinstance(opt.gpu_ids, list) and len(opt.gpu_ids) > 0 and opt.gpu_ids[0] < 0:
    opt.gpu_ids = []
# override some options for our inference use-case
opt.model = 'IBCLN'
opt.checkpoints_dir = './checkpoints'
# Use the experiment name under which the pretrained model is stored
# The default `name` is 'experiment_name' so make sure it matches the folder name
opt.name = 'IBCLN'
opt.epoch = 'final'  # matches the filename in checkpoints/IBCLN
opt.batch_size = 1
opt.serial_batches = True
opt.no_flip = True
opt.num_test = 1
opt.eval = True
opt.isTrain = False

model = create_model(opt)
model.setup(opt)
if opt.eval:
    model.eval()

# simple image preprocessing: chỉ chuyển về tensor, không resize
preprocess = transforms.ToTensor()


def run_model(input_tensor, image_path):
    """Run the loaded IBCLN model on a single input tensor."""
    data = {
        'B_paths': [image_path],
        'I': input_tensor,
        'T': input_tensor,
    }
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    # Xuất ra real_T (ảnh đã xóa bóng mờ)
    return visuals.get('real_T')


@app.route('/', methods=['GET', 'POST'])
def index():
    input_url = None
    result_url = None
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            filename = file.filename
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            
            # prepare input for model
            img = Image.open(upload_path).convert('RGB')
            input_tensor = preprocess(img).unsqueeze(0).to(model.device)
            output_tensor = run_model(input_tensor, upload_path)
            
            # convert output tensor to image and save
            output_np = util.tensor2im(output_tensor)
            out_img = Image.fromarray(output_np)
            result_filename = 'result_' + filename
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            out_img.save(result_path)

            input_url = '/' + upload_path.replace('\\', '/')
            result_url = '/' + result_path.replace('\\', '/')
    return render_template('index.html', input_image=input_url, result_image=result_url)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
