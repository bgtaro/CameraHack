import cv2
import numpy as np

# --- 設定 ---
CAMERA_INDEX = 0 
# ダウンロードした顔検出XMLファイルへのパス
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
# 笑い男のロゴ画像ファイルへのパス
LOGO_PATH = "laughing_man_logo.png"

# --- 透過度付き合成関数（前回のコードから流用） ---
def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """
    背景画像(img)の指定位置(pos)に、透過マスク(alpha_mask)を使って前景画像(img_overlay)を合成します。
    """
    x, y = pos
    h_overlay, w_overlay, _ = img_overlay.shape
    
    # オーバーレイする領域を計算
    y1, y2 = max(0, y), min(img.shape[0], y + h_overlay)
    x1, x2 = max(0, x), min(img.shape[1], x + w_overlay)
    
    # 合成する画像サイズを調整
    img_crop = img[y1:y2, x1:x2]
    
    # ロゴとマスクも切り取り（オーバーレイ領域に合わせる）
    logo_crop = img_overlay[0:y2-y1, 0:x2-x1]
    alpha_crop = alpha_mask[0:y2-y1, 0:x2-x1]

    # アルファチャンネルを正規化
    alpha = alpha_crop / 255.0
    alpha_inv = 1.0 - alpha
    
    # リアルタイム合成 (ブレンド) 処理
    for c in range(0, 3):
        img_crop[:, :, c] = (img_crop[:, :, c] * alpha_inv) + \
                             (logo_crop[:, :, c] * alpha)
    
    return img

# --- 初期化 ---
try:
    # 1. 顔検出器のロード
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        raise FileNotFoundError(f"カスケードファイルが見つかりません: {FACE_CASCADE_PATH}")

    # 2. ロゴ画像のロード（アルファチャンネル付き）
    logo_img = cv2.imread(LOGO_PATH, cv2.IMREAD_UNCHANGED)
    if logo_img is None:
        raise FileNotFoundError(f"ロゴファイルが見つかりません: {LOGO_PATH}")
        
    # ロゴから色情報と透過情報を分離
    logo_color = logo_img[:, :, :3]  
    logo_alpha = logo_img[:, :, 3]   

except FileNotFoundError as e:
    print(f"致命的なエラー: {e}")
    print("必要なファイル（haarcascade_frontalface_default.xml または laughing_man_logo.png）を確認してください。")
    exit()

# 3. カメラの初期化
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"エラー: カメラ({CAMERA_INDEX})を開けませんでした。")
    exit()

print("--- 📺 笑い男 顔トラッキング＆上書き開始 ---")
print("Qキーを押すと終了します。")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("フレームの取得に失敗しました。")
        break

    # 4. 顔検出処理
    # 処理速度向上のため、画像をモノクロに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 顔を検出 (scaleFactor=1.3, minNeighbors=5 は検出の厳しさのパラメーター)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=2,
        minSize=(50, 50)
    )

    frame_hacked = frame.copy()

    # 5. 検出した顔にロゴを上書き
    for (x, y, w, h) in faces:
        # 検出した顔のサイズ(w, h)に合わせてロゴをリサイズ
        resized_logo_color = cv2.resize(logo_color, (w, h), interpolation=cv2.INTER_AREA)
        resized_logo_alpha = cv2.resize(logo_alpha, (w, h), interpolation=cv2.INTER_AREA)
        
        # 合成関数を呼び出し、顔の位置(x, y)に上書き
        frame_hacked = overlay_image_alpha(
            frame_hacked, 
            resized_logo_color, 
            (x, y), 
            resized_logo_alpha
        )
        
        # 検出された顔の周りに赤い枠線（デバッグ用、最終的には削除可）
        # cv2.rectangle(frame_hacked, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # 6. 結果の表示
    cv2.imshow('Laughing Man Face Hack', frame_hacked)
    
    # 'q' キーが押されたら終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 終了処理
cap.release()
cv2.destroyAllWindows()
print("--- 処理を終了しました ---")