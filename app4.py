import cv2
import time
# https://qiita.com/seri28/items/3ae4a2c87e352e976b46
#movie = cv2.VideoCapture('./movie/park.mp4')
cap = cv2.VideoCapture(0)


red = (0, 0, 255) # 枠線の色
before = None # 前回の画像を保存する変数
fps = int(cap.get(cv2.CAP_PROP_FPS)) #動画のFPSを取得

while True:
    # 画像を取得
    ret, frame = cap.read()
    # 再生が終了したらループを抜ける
    if ret == False: break

    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # BGRのチャンネル並びをRGBの並びに変更(matplotlibで結果を表示するため)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    # 色相の範囲を指定
    lower_hue = 16  # 下限
    upper_hue = 20  # 上限
    # 色相に基づいて閾値処理を行う
    mask = cv2.inRange(hsv_image, (lower_hue, 50, 50), (upper_hue, 255, 255))
    # 元画像とマスクを結合して結果を得る
    #result = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)


    # 白黒画像に変換
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if before is None:
        before = mask.astype("float")
        continue
    #現在のフレームと移動平均との差を計算
    cv2.accumulateWeighted(mask, before, 0.8)
    frameDelta = cv2.absdiff(mask, cv2.convertScaleAbs(before))
    #frameDeltaの画像を２値化
    thresh = cv2.threshold(frameDelta, 3, 255, cv2.THRESH_BINARY)[1]
    #輪郭のデータを得る
    contours = cv2.findContours(thresh,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)[0]

    # 差分があった点を画面に描く
    for target in contours:
        x, y, w, h = cv2.boundingRect(target)
        if w < 30: continue # 小さな変更点は無視
        cv2.rectangle(frame, (x, y), (x+w, y+h), red, 2)

    #ウィンドウでの再生速度を元動画と合わせる
    time.sleep(1/fps)
    # ウィンドウで表示
    cv2.imshow('target_frame', frame)
    # Enterキーが押されたらループを抜ける
    if cv2.waitKey(1) == 13: break

cv2.destroyAllWindows() # ウィンドウを破棄