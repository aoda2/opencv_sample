import cv2

# https://qiita.com/KMiura95/items/4eed79a7da6b3dafa96d
# ファイルを使う場合
# filepath = "vtest.avi"
# cap = cv2.VideoCapture(filepath)
# Webカメラを使うときはこちら
cap = cv2.VideoCapture(0)

avg = None

while True:
    # 1フレームずつ取得する。
    ret, frame = cap.read()
    if not ret:
        break

    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # BGRのチャンネル並びをRGBの並びに変更(matplotlibで結果を表示するため)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    # 色相の範囲を指定
    lower_hue = 17  # 下限
    upper_hue = 18  # 上限
    # 色相に基づいて閾値処理を行う
    gray = cv2.inRange(hsv_image, (lower_hue, 50, 50), (upper_hue, 255, 255))

    # グレースケールに変換
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 比較用のフレームを取得する
    if avg is None:
        avg = gray.copy().astype("float")
        continue

    # 現在のフレームと移動平均との差を計算
    cv2.accumulateWeighted(gray, avg, 0.6)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    # デルタ画像を閾値処理を行う
    thresh = cv2.threshold(frameDelta, 3, 255, cv2.THRESH_BINARY)[1]
    # 画像の閾値に輪郭線を入れる
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    # 結果を出力
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()