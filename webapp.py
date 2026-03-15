import os
import sys
import torch
from flask import Flask, render_template, request
from PIL import Image
import torchvision.transforms as transforms

# Import từ source code project
from options.test_options import TestOptions
from models import create_model
from util import util
from data.base_dataset import get_params, get_transform

# Cấu hình thư mục
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

def setup_options():
    sys.argv = [
        'webapp.py',
        '--dataroot', './static/uploads', 
        '--name', 'IBCLN',      
        '--model', 'IBCLN',     
        '--dataset_mode', 'resize_natural_3',
        '--gpu_ids', '-1',      
        '--no_flip',            
        '--epoch', 'final',     
        '--eval',               
    ]
    test_options = TestOptions()
    opt = test_options.parse()
    opt.isTrain = False        
    opt.batch_size = 1         
    opt.serial_batches = True  
    opt.eval = True            
    opt.num_test = 1           
    opt.results_dir = './static/results'
    return opt

opt = setup_options()
# Tự động nhận diện thiết bị (MPS cho Mac, CUDA cho PC)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
if torch.cuda.is_available(): device = torch.device('cuda')

model = create_model(opt)
model.setup(opt)
model.eval()

def run_inference(img_path):
    img_pil = Image.open(img_path).convert('RGB')
    orig_size = img_pil.size
    
    # --- TUNING 1: ĐỒNG BỘ DATA PIPELINE ---
    # Thay vì dùng transforms.Resize thủ công, ta dùng get_transform của project
    # để đảm bảo dải màu và cách căn lề giống hệt lúc training.
    params = get_params(opt, img_pil.size)
    transform = get_transform(opt, params, grayscale=(opt.input_nc == 1))
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # --- TUNING 2: ĐỊNH NGHĨA DATA CHUẨN ---
    data = {
        'I': input_tensor,
        'T': input_tensor,      # Khởi tạo Transmission bằng ảnh gốc
        'B_paths': [img_path],  # Fix lỗi B_paths
        'A_paths': [img_path],
        'isNatural': torch.tensor([1]).to(device)
    }

    model.set_input(data)
    
    with torch.no_grad():
        # --- TUNING 3: ÉP SỐ BƯỚC LẶP (CRITICAL) ---
        # IBCLN cần lặp nhiều lần để xóa mờ sạch. Thử chỉnh n_iters từ 3-8.
        if hasattr(model, 'n_iters'):
            model.n_iters = 3 
        
        # Nếu model có hàm init_state, phải gọi nó trước khi test
        if hasattr(model, 'init_state'):
            model.init_state()
            
        model.test() 
    
    # --- TUNING 4: TRÍCH XUẤT ẢNH THÔNG MINH ---
    visuals = model.get_current_visuals()
    
    # Ưu tiên lấy fake_Ts (danh sách ảnh qua các vòng lặp)
    output_tensor = None
    if hasattr(model, 'fake_Ts') and len(model.fake_Ts) > 0:
        output_tensor = model.fake_Ts[-1] # Lấy kết quả cuối cùng (sạch nhất)
    else:
        # Nếu không có attribute, tìm trong dictionary visuals
        for key in ['fake_T', 'fake_B', 'fake']:
            if key in visuals:
                output_tensor = visuals[key]
                break

    if output_tensor is None:
        raise ValueError(f"Model không sinh ra kết quả. Các key hiện có: {list(visuals.keys())}")

    # Hậu xử lý chuẩn của project
    output_np = util.tensor2im(output_tensor)
    out_img = Image.fromarray(output_np)
    return out_img.resize(orig_size, Image.LANCZOS)

@app.route('/', methods=['GET', 'POST'])
def index():
    input_url = None
    result_url = None
    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename != '':
            filename = file.filename
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            try:
                out_img = run_inference(upload_path)
                result_filename = 'cleared_' + filename
                result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
                out_img.save(result_path)

                # Sử dụng đường dẫn tương đối cho Flask
                input_url = f"uploads/{filename}"
                result_url = f"results/{result_filename}"
            except Exception as e:
                print(f"❌ Error: {e}")
                
    return render_template('index.html', input_image=input_url, result_image=result_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)