# Pet Classifier

Nhận diện giống chó/mèo bằng TensorFlow + FastAPI.

## Cài đặt

```bash
# Tạo môi trường ảo
python -m venv venv
.\venv\Scripts\activate  # Windows

# Cài thư viện
pip install -r requirements.txt

# Chạy server
python main.py
```

Mở http://localhost:8000 để sử dụng giao diện web.

## API Endpoints

| Method | Path        | Mô tả                     |
|--------|-------------|---------------------------|
| GET    | `/`         | Giao diện web             |
| POST   | `/predict`  | Nhận diện ảnh (file)      |
| GET    | `/health`   | Kiểm tra server           |