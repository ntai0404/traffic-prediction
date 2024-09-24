# Traffic Prediction Project

## Giới thiệu
Dự án này sử dụng các thuật toán học máy để dự đoán tình trạng giao thông dựa trên các yếu tố như ô nhiễm không khí, độ ẩm, tốc độ gió, và nhiều yếu tố khác.

## Yêu cầu
Trước khi bắt đầu, hãy đảm bảo rằng bạn đã cài đặt Python 3.6 hoặc cao hơn.

## Cài đặt

1. **Clone dự án**:
   ```bash
   git clone https://github.com/yourusername/traffic_prediction.git
   
   gõ trong terminal thư mục gốc của dự án:
   cd traffic_prediction 


2.Tạo môi trg ảo:
pip install virtualenv // lần thứ 2 trở đi ko làm
python -m venv venv1 // lần thứ 2 trở đi ko làm

3.kích hoạt môi trg ảo:
venv1\Scripts\activate

4. cài các gói cần thiết:// lần thứ 2 trở đi ko làm
pip install -r requirements.txt
 pip install scikit-learn
pip install pandas  
 pip install flask  


5. khỏi động dự án ;khởi chạy dc luôn vì đã có file pkl từ việc huấn luyện các modul trong src/ 

python web/app.py

6. Sau khi thấy hiện ra đường dẫn -> gõ lên trình duyệt để chạy.

7. Đóng dự án: 
trong terminal đang bật dự án:
ctrl c

8. đóng máy ảo venv:
trong terminal :
deactivate
