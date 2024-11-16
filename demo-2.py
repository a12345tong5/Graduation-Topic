import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import keras

# 載入模型
model = keras.models.load_model('./model.h5')

# 定義分類類別名稱
class_names = {0: '5', 1: '3', 2: '1', 3: '4', 4: '0', 5: '2'}

# 初始化 GUI 視窗
window = tk.Tk()
window.title("門禁系統 - 人臉辨識")
window.geometry("300x500")
window.configure(bg='#2E4053')

# 標題
title_label = tk.Label(window, text="門禁系統", font=("Arial", 24, "bold"), bg='#2E4053', fg='white')
title_label.pack(pady=20)

# 顯示選取的圖片區域
image_frame = tk.Frame(window, bg='#34495E', width=200, height=200)
image_frame.pack(pady=20)
image_frame.pack_propagate(False)
image_label = tk.Label(image_frame, bg='#34495E')
image_label.pack(expand=True)

# 顯示預測結果
result_text = tk.StringVar()
result_label = tk.Label(window, textvariable=result_text, font=("Arial", 16), bg='#2E4053', fg='white')
result_label.pack(pady=20)

def load_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        messagebox.showerror("錯誤", "請選擇圖片！")
        return

    # 顯示圖片
    im = Image.open(file_path).resize((120, 120))
    im_tk = ImageTk.PhotoImage(im)
    image_label.configure(image=im_tk)
    image_label.image = im_tk  # 保持引用避免被垃圾回收

    # 預處理圖片
    image_array = np.asarray(im.resize((120, 120))).astype(np.float32) / 255.0
    X = np.expand_dims(image_array, axis=0)

    # 預測
    prediction = model.predict(X)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # 判斷結果
    if predicted_class == 0:  # 假設0代表"不通過"
        result_text.set("認證結果：不通過")
        result_label.config(fg='red')
    else:
        result_text.set(f"認證結果：通過 - {class_names[predicted_class]}")
        result_label.config(fg='green')

# 按鈕
load_button = tk.Button(window, text="選擇圖片進行辨識", command=load_and_predict, font=("Arial", 20),
                        bg='#1ABC9C', fg='white', width=16)
load_button.place(x=18, y=400)
#load_button.pack(pady=20)

# 主循環
window.mainloop()
