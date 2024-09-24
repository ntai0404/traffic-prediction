import random
import pandas as pd

# Giả lập dữ liệu cho 1000 mẫu loại 1 (Thông thoáng)
data_type1 = {
    'is_holiday': [random.choice([0, 1]) for _ in range(1000)],
    'air_pollution_index': [round(random.uniform(20, 80), 2) for _ in range(1000)],  # Chỉ số ô nhiễm không khí
    'temperature': [round(random.uniform(20, 30), 2) for _ in range(1000)],  # Nhiệt độ
    'rain_p_h': [round(random.uniform(0, 2), 2) for _ in range(1000)],  # Mưa
    'visibility_in_miles': [round(random.uniform(5, 10), 2) for _ in range(1000)],  # Tầm nhìn
    'traffic_condition': [0] * 1000  # Phân loại là 1 (Thông thoáng)
}

# Tạo DataFrame cho loại 1
df_type1 = pd.DataFrame(data_type1)
df_type1.to_csv('./data/traffic_data.csv', index=False)

# Giả lập dữ liệu cho 1000 mẫu loại 2 (Đông đúc)
data_type2 = {
    'is_holiday': [random.choice([0, 1]) for _ in range(1000)],
    'air_pollution_index': [round(random.uniform(10, 50), 2) for _ in range(1000)],  # Chỉ số ô nhiễm không khí
    'temperature': [round(random.uniform(20, 25), 2) for _ in range(1000)],  # Nhiệt độ
    'rain_p_h': [round(random.uniform(0, 0.5), 2) for _ in range(1000)],  # Mưa
    'visibility_in_miles': [round(random.uniform(5, 10), 2) for _ in range(1000)],  # Tầm nhìn
    'traffic_condition': [1] * 1000  # Phân loại là 2 (Đông đúc)
}

# Tạo DataFrame cho loại 2
df_type2 = pd.DataFrame(data_type2)
df_type2.to_csv('./data/traffic_data.csv', index=False)

# Giả lập dữ liệu cho 1000 mẫu loại 3 (Ùn tắc)
data_type3 = {
    'is_holiday': [random.choice([0, 1]) for _ in range(1000)],
    'air_pollution_index': [round(random.uniform(30, 80), 2) for _ in range(1000)],  # Chỉ số ô nhiễm không khí
    'temperature': [round(random.uniform(25, 35), 2) for _ in range(1000)],  # Nhiệt độ
    'rain_p_h': [round(random.uniform(0.5, 2.0), 2) for _ in range(1000)],  # Mưa
    'visibility_in_miles': [round(random.uniform(2, 5), 2) for _ in range(1000)],  # Tầm nhìn
    'traffic_condition': [2] * 1000  # Phân loại là 3 (Ùn tắc)
}

# Tạo DataFrame cho loại 3
df_type3 = pd.DataFrame(data_type3)
df_type3.to_csv('./data/traffic_data.csv', index=False)
