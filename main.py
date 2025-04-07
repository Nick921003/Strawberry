from ultralytics import YOLO
from multiprocessing import freeze_support
import torch
import cv2
import os
import numpy as np

def resize_for_display(img, max_width=1280, max_height=720):
    """
    如果圖片尺寸超過最大限制，則按比例調整圖片大小以便顯示。

    Args:
        img (np.ndarray): 輸入圖片 (NumPy array)。
        max_width (int): 顯示的最大寬度。
        max_height (int): 顯示的最大高度。

    Returns:
        np.ndarray: 調整大小後的圖片（如果需要）或原始圖片。
        float: 實際應用的縮放比例（如果未縮放則為 1.0）。
    """
    h, w = img.shape[:2]
    scale = 1.0 # 預設縮放比例

    # 檢查是否需要縮放
    if h > max_height or w > max_width:
        scale_h = max_height / h
        scale_w = max_width / w
        scale = min(scale_h, scale_w) # 計算限制性縮放比例

    # 如果計算出的比例小於1，則執行縮放
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        # 使用 cv2.resize 進行縮放
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"圖片尺寸 ({w}x{h}) 過大，已縮放至 {new_w}x{new_h} (比例: {scale:.2f})")
        return resized_img, scale
    else:
        # 如果不需要縮放，返回原始圖片和比例 1.0
        return img, 1.0
    
def detect_and_show_masks(model_path, image_path, confidence_threshold=0.7, max_display_width=1280, max_display_height=720):
    """
    載入 YOLO 分割模型並對指定圖片進行分割，
    在可調整大小的視窗中顯示信心度大於指定閾值的結果，
    並確保初始顯示尺寸按比例縮放至不超過指定大小（使用獨立函式）。

    Args:
        model_path (str): 模型路徑.
        image_path (str): 圖片路徑.
        confidence_threshold (float, optional): 信心閾值. Defaults to 0.7.
        max_display_width (int, optional): 顯示圖片的最大寬度. Defaults to 1280.
        max_display_height (int, optional): 顯示圖片的最大高度. Defaults to 720.

    Returns:
        results: YOLO 模型的原始偵測結果 (如果有的話).
        bool: 是否成功執行.
    """
    window_title = "Segmentation Result"
    try:
        if not os.path.exists(model_path): raise FileNotFoundError(f"模型檔案未找到: {model_path}")
        if not os.path.exists(image_path): raise FileNotFoundError(f"圖片檔案未找到: {image_path}")

        model = YOLO(model_path)
        original_img = cv2.imread(image_path)
        if original_img is None: raise ValueError(f"無法讀取圖片: {image_path}")

        results = model(original_img, conf=confidence_threshold)

        display_img_source = None # 用於調整大小前的來源圖片

        # 根據是否有結果決定顯示標註圖或原圖
        valid_results_exist = results and results[0].boxes is not None and len(results[0].boxes) > 0
        if not valid_results_exist:
            print(f"模型未偵測到任何信心度超過 {confidence_threshold} 的結果。顯示原圖。")
            display_img_source = original_img.copy()
            window_title = "Original Image (No Detections)"
        else:
            annotated_img = results[0].plot()
            display_img_source = annotated_img
            window_title = f"Segmentation Result (Conf > {confidence_threshold})"

        # *** 使用獨立的 resize_for_display 函式來調整圖片大小 ***
        display_img_final, applied_scale = resize_for_display(
            display_img_source,
            max_display_width,
            max_display_height
        )

        # 創建可調整大小的視窗並顯示最終圖片
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.imshow(window_title, display_img_final)
        print("按任意鍵關閉結果視窗...")
        cv2.waitKey(0)

        return_results = results if valid_results_exist else None
        return return_results, True

    except Exception as e:
        print(f"偵測過程發生錯誤: {str(e)}")
        return None, False
    finally:
        # 確保無論如何都關閉視窗
        cv2.destroyAllWindows()

def train_segmentation_model(yaml_path="data.yaml", epochs=100, batch_size=8, img_size=640, workers=2, save_best=True):
    """
    使用 YOLOv11 模型進行圖像分割任務的訓練。
    Args:
        yaml_path (str, optional): 資料集設定檔 (YAML 檔案) 的路徑。預設為 "data.yaml"。
        epochs (int, optional): 訓練的總 epoch 數。預設為 100。
        batch_size (int, optional): 每個批次的圖像數量。預設為 8。
        img_size (int, optional): 輸入圖像的尺寸 (像素)。預設為 640。
        workers (int, optional): 資料載入的線程數。預設為 2。
        save_best (bool, optional): 是否儲存最佳模型。預設為 True。
    """
    try:
        # 檢查 CUDA 是否可用
        if not torch.cuda.is_available():
            print("CUDA is not available. Training on CPU instead.")
            device = "cpu"
        else:
            device = "cuda"

        # 載入 YOLOv11 分割預訓練模型
        model = YOLO("best.pt")  # Changed to segmentation model

        # 訓練分割模型
        model.train(
            data=yaml_path,      # 資料集設定
            task="segment",        # 設定任務為分割
            epochs=epochs,             # 訓練 epoch 數
            batch=batch_size,               # 批次大小
            imgsz=img_size,             # 圖像尺寸
            workers=workers,             # 資料載入線程數
            device=device,         # 使用 GPU 進行訓練
            save=save_best,             # 儲存最佳模型
        )

    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    freeze_support()  # 用於 Windows 系統的多進程支持
    # 設定你的 YAML 檔案路徑 (如果不是預設的 "data.yaml")
    # 調用訓練函式，可以根據需要調整參數
    # train_segmentation_model("data4.yaml", 150, 16, 640, 4, True)
    model_path = "runs/segment/train28/weights/best.pt"
    image_path = "C:/Users/pjw92/Desktop/IMG_2283.png"
    results_data, success = detect_and_show_masks(
        model_path,
        image_path,
        confidence_threshold=0.7,
        max_display_width=4032, 
        max_display_height=3024  
    )
    
    if success:
        print("偵測成功，結果已顯示。")
    else:
        print("偵測失敗。")

