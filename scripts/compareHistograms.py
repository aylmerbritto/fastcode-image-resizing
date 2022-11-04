import cv2

path1 = "/afs/ece.cmu.edu/usr/arexhari/Public/645-project/inputs/640x480.jpg"
path2 = "/afs/ece.cmu.edu/usr/arexhari/Public/645-project/inputs/640x480.jpg"
img1 = cv2.imread(path1)
img2 = cv2.imread(path2)
result = [0,0,0,0,0]
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED',     'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
for i in range(5):
    result[i] = cv2.matchTemplate(img1,img2,i)
    print ("Method {}  : Result{}") .format(methods[i],result[i])