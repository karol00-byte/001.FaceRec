import sys, os, dlib, glob, numpy
from skimage import io
from sys import argv
import cv2
import imutils
import face_recognition
current_path = os.getcwd()

predictor_path = ".\\03.pic\\shape_predictor_68_face_landmarks.dat\\shape_predictor_68_face_landmarks.dat"

#人臉辨識模型路徑
face_rec_model_path = ".\\03.pic\\dlib_face_recognition_resnet_model_v1.dat\\dlib_face_recognition_resnet_model_v1.dat"
#比對人臉圖片資料夾名稱
faces_folder_path=".\\03.pic\\rec"

#載入人臉辨識器
detector=dlib.get_frontal_face_detector()
#載入人臉特徵檢測器
sp = dlib.shape_predictor(predictor_path)
#載入人臉辨識檢視器
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
#比對人臉描述子列表
descriptors = []
#比對人臉名稱列表
candidate = []

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
   # print("Processing file: {}".format(f))
    base = os.path.basename(f)
    # 依序取得人名
    candidate.append(os.path.splitext(base)[0])
    img = io.imread(f)
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        v = numpy.array(face_descriptor)
        descriptors.append(v)

imgs =["anna1","anna2","emma1","emma2","lanlan-1"]
qq=0

for x in imgs:
    img_name =x+".jpg"
    img = io.imread(img_name)
    #人臉偵測
    dets = detector(img, 1)
    dist = []
    for k, d in enumerate(dets):
        #dist[]
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        d_test = numpy.array(face_descriptor)

        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()

        # 以方框標示偵測人臉
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

        for i in descriptors:
            dist_ = numpy.linalg.norm(i - d_test)
            dist.append(dist_)

        c_d = dict(zip(candidate, dist))
        cd_sorted = sorted(c_d.items(), key=lambda d:d[1])
        rec_name = cd_sorted[0][0]
        print("result={}".format(rec_name))

        cv2.putText(img, rec_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255)
                , 2, cv2.LINE_AA)

    img = imutils.resize(img, width=600)
    qq=qq+1
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("outcome{}".format(qq), img)

# 隨意key一鍵結束
cv2.waitKey(0)
cv2.destroyAllWindows()