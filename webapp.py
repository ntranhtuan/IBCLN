"""
WEBAPP.PY - Phiên bản đơn giản cho IBCLN
Tương thích với file index.html mới
"""

import os
import sys
import torch
from flask import Flask, render_template, request, send_file
from PIL import Image
import torchvision.transforms as transforms
import time

# Import từ source code project
from options.test_options import TestOptions
from models import create_model
from util import util
from data.base_dataset import get_params, get_transform

# ==================== CẤU HÌNH ====================
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# ==================== CẤU HÌNH OPTIONS ====================
def setup_options():
    """Thiết lập options với checkpoint final"""
    
    sys.argv = [
        'webapp.py',
        '--dataroot', './static/uploads', 
        '--name', 'IBCLN',      
        '--model', 'IBCLN',     
        '--dataset_mode', 'resize_natural_3',
        '--gpu_ids', '-1',      # CPU mode
        '--no_flip',            
        '--epoch', 'final',     # Dùng checkpoint final
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

# ==================== KHỞI TẠO MODEL ====================
opt = setup_options()

# Tự động nhận diện thiết bị (MPS cho Mac, CUDA cho PC)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
if torch.cuda.is_available(): 
    device = torch.device('cuda')
    opt.gpu_ids = [0]  # Bật GPU

print(f"🚀 Using device: {device}")

model = create_model(opt)
model.setup(opt)
model.eval()
print("✅ Model loaded successfully!")

# ==================== HÀM INFERENCE ====================
def run_inference(img_path, n_iters=3):
    """
    Chạy inference với số bước lặp tùy chỉnh
    Args:
        img_path: đường dẫn ảnh
        n_iters: số bước lặp (mặc định 3)
    """
    start_time = time.time()
    
    img_pil = Image.open(img_path).convert('RGB')
    orig_size = img_pil.size
    print(f"🖼️ Original size: {orig_size}")
    
    # Đồng bộ data pipeline
    params = get_params(opt, img_pil.size)
    transform = get_transform(opt, params, grayscale=(opt.input_nc == 1))
    input_tensor = transform(img_pil).unsqueeze(0).to(device)
    print(f"📊 Input tensor shape: {input_tensor.shape}")

    # Định nghĩa data chuẩn
    data = {
        'I': input_tensor,
        'T': input_tensor,      # Khởi tạo Transmission bằng ảnh gốc
        'B_paths': [img_path],
        'A_paths': [img_path],
        'isNatural': torch.tensor([1]).to(device)
    }

    model.set_input(data)
    
    with torch.no_grad():
        # Ép số bước lặp
        if hasattr(model, 'n_iters'):
            model.n_iters = n_iters
            print(f"🔄 Running with n_iters = {n_iters}")
        
        # Reset state LSTM nếu có
        if hasattr(model, 'init_state'):
            model.init_state()
            
        model.test() 
    
    # Trích xuất ảnh kết quả
    visuals = model.get_current_visuals()
    
    output_tensor = None
    if hasattr(model, 'fake_Ts') and len(model.fake_Ts) > 0:
        output_tensor = model.fake_Ts[-1]  # Lấy kết quả cuối cùng
        print(f"📸 Using fake_Ts[-1], shape: {output_tensor.shape}")
    else:
        for key in ['fake_T', 'fake_B', 'fake']:
            if key in visuals:
                output_tensor = visuals[key]
                print(f"📸 Using {key}")
                break

    if output_tensor is None:
        raise ValueError(f"Model không sinh ra kết quả. Các key: {list(visuals.keys())}")

    # Hậu xử lý
    output_np = util.tensor2im(output_tensor)
    out_img = Image.fromarray(output_np)
    out_img = out_img.resize(orig_size, Image.LANCZOS)
    
    elapsed = time.time() - start_time
    print(f"⏱️ Inference time: {elapsed:.2f}s")
    
    return out_img

# ==================== ROUTES ====================
@app.route('/', methods=['GET', 'POST'])
def index():
    input_url = None
    result_url = None
    processing_time = None
    n_iters_used = 3
    
    if request.method == 'POST':
        file = request.files.get('image')
        
        # Lấy số bước lặp từ form (nếu có)
        try:
            n_iters_used = int(request.form.get('n_iters', 3))
        except:
            n_iters_used = 3
        
        if file and file.filename != '':
            # Kiểm tra định dạng file
            ext = file.filename.rsplit('.', 1)[-1].lower()
            if ext not in ['jpg', 'jpeg', 'png']:
                return "Chỉ hỗ trợ JPG/PNG", 400
            
            # Tạo filename an toàn
            timestamp = int(time.time())
            filename = f"{timestamp}_{file.filename}"
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            
            try:
                print(f"\n{'='*50}")
                print(f"🚀 Processing: {filename}")
                
                # Chạy inference
                start_time = time.time()
                out_img = run_inference(upload_path, n_iters=n_iters_used)
                processing_time = time.time() - start_time
                
                # Lưu kết quả
                result_filename = f"cleared_{filename}"
                result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
                out_img.save(result_path, quality=95, optimize=True)

                input_url = f"uploads/{filename}"
                result_url = f"results/{result_filename}"
                
                print(f"✅ Done: {result_filename}")
                print('='*50)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                return f"Lỗi xử lý: {str(e)}", 500
                
    return render_template('index.html', 
                         input_image=input_url, 
                         result_image=result_url,
                         processing_time=processing_time,
                         n_iters_used=n_iters_used)

@app.route('/download/<path:filename>')
def download_file(filename):
    """Tải ảnh kết quả về máy"""
    try:
        safe_filename = os.path.basename(filename)
        file_path = os.path.join(RESULT_FOLDER, safe_filename)
        
        if not os.path.exists(file_path):
            return "File không tồn tại", 404
            
        return send_file(file_path, as_attachment=True, download_name=safe_filename)
    except Exception as e:
        return str(e), 500

# ==================== MAIN ====================
if __name__ == '__main__':
    print(f"""
    🚀 IBCLN WebApp - Simple Version
    {'='*50}
    Device: {device}
    Checkpoint: final
    {'='*50}
    """)
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)