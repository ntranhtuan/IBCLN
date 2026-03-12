import os
import sys
import torch
from flask import Flask, render_template, request
from PIL import Image
import torchvision.transforms as transforms

from options.test_options import TestOptions
from models import create_model
from util import util

############################################
# CONFIG
############################################

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

############################################
# FLASK APP
############################################

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER


############################################
# FAKE CLI OPTIONS FOR IBCLN
############################################

sys.argv = [
    sys.argv[0],
    "--dataroot", ".",
    "--model", "IBCLN",
    "--name", "IBCLN",
    "--dataset_mode", "resize_natural_3",
    "--gpu_ids", "-1"
]

############################################
# LOAD MODEL
############################################

opt = TestOptions().parse()

opt.checkpoints_dir = "./checkpoints"
opt.epoch = "final"
opt.batch_size = 1
opt.serial_batches = True
opt.no_flip = True
opt.num_test = 1
opt.eval = True
opt.isTrain = False

model = create_model(opt)
model.setup(opt)
model.eval()

print("Model loaded from:", opt.checkpoints_dir)
print("Epoch:", opt.epoch)
print("Device:", model.device)

############################################
# IMAGE PREPROCESS
############################################

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

############################################
# INFERENCE FUNCTION
############################################

def run_model(input_tensor, image_path):

    data = {
        "B_paths": [image_path],
        "I": input_tensor,
        "T": input_tensor,
        "isNatural": 1
    }

    model.set_input(data)

    with torch.no_grad():

        # reset LSTM state
        model.init()

        model.forward()

    print("cascade outputs:", len(model.fake_Ts))

    output_tensor = model.fake_Ts[-1]

    return output_tensor


############################################
# WEB ROUTE
############################################

@app.route("/", methods=["GET", "POST"])
def index():

    input_url = None
    result_url = None

    if request.method == "POST":

        file = request.files.get("image")

        if file:

            filename = file.filename

            upload_path = os.path.join(
                app.config["UPLOAD_FOLDER"], filename)

            file.save(upload_path)

            ###################################
            # LOAD IMAGE
            ###################################

            img = Image.open(upload_path).convert("RGB")

            original_size = img.size

            ###################################
            # PREPROCESS
            ###################################

            input_tensor = preprocess(img).unsqueeze(0).to(model.device)

            print("input shape:", input_tensor.shape)

            ###################################
            # RUN MODEL
            ###################################

            output_tensor = run_model(input_tensor, upload_path)

            ###################################
            # DEBUG
            ###################################

            diff = torch.mean(torch.abs(output_tensor - input_tensor))
            print("diff:", diff.item())

            ###################################
            # SAVE RESULT
            ###################################

            output_np = util.tensor2im(output_tensor.detach())

            out_img = Image.fromarray(output_np)

            # resize về kích thước ban đầu
            out_img = out_img.resize(original_size, Image.BICUBIC)

            result_filename = "result_" + filename

            result_path = os.path.join(
                app.config["RESULT_FOLDER"], result_filename)

            out_img.save(result_path)

            ###################################
            # URL
            ###################################

            input_url = "/" + upload_path.replace("\\", "/")
            result_url = "/" + result_path.replace("\\", "/")

    return render_template(
        "index.html",
        input_image=input_url,
        result_image=result_url
    )


############################################
# RUN SERVER
############################################

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True
    )