import sys
#from time import sleep
# https://greenhornprofessional.hatenablog.com/entry/2021/03/25/230313
import cv2
#Webカメラのインスタンス作成
#引数はカメラ番号。0番はラップトップ内蔵カメラ。
cap = cv2.VideoCapture(0)
#Webカメラのインスタンス作成に失敗したらプログラム終了
if not cap.isOpened():
    print("Camera Not Found")
    sys.exit()
#キャプチャと円検出を繰り返す
while True:
#    sleep(1)
    #キャプチャ取得
    _, frame = cap.read()
    #前処理
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray ,7)
    gray = cv2.equalizeHist(gray)
    #円検出
    circles = cv2.HoughCircles(gray,
                               cv2.HOUGH_GRADIENT,
                               dp=4,
                               minDist=50,
                               param1=400,
                               param2=100,
                               minRadius=30,
                               maxRadius=50)
    #円の検出に成功したらオリジナルのimgに円を描画
    if circles is not None:
        for i in circles[0].astype('uint16'):
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
    cv2.imshow('Detect', frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
cap.release()