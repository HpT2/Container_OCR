import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from imutils import contours

#regconize character
def ocr(code_reg):
    labels = [
    '0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G',
    'H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'
    ]

    #load pretrained model
    MODEL_PATH = './EfficientNetB0/trained_model'
    model = load_model(MODEL_PATH, compile=False)
    code_reg = cv2.cvtColor(code_reg,cv2.COLOR_BGR2GRAY)
    _, code_reg = cv2.threshold(code_reg, 220, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    if code_reg[0,0] == 0:
        code_reg = ~code_reg
        
    bw = code_reg.copy()
    height ,width = code_reg.shape
    
    #detect character
    mask = cv2.blur(code_reg,(3,3)) 
    mask = cv2.Canny(code_reg, 120, 255)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts,_ = contours.sort_contours(cnts)

    #predict character
    for idx in range(len(cnts)):
        x, y, w, h = cv2.boundingRect(cnts[idx])
        #remove box to small or to large or near edge of image
        if  w > 0.1*width and h > 0.1*height and w < 100 and h < 100:
            crop_img = bw[y:y + h , x:x + w ]
            crop_img = cv2.cvtColor(crop_img,cv2.COLOR_GRAY2BGR)
    
            # create new image of desired size and color for padding
            old_image_height, old_image_width, channels = crop_img.shape
            new_image_width = old_image_width + 20
            new_image_height = old_image_height + 40
            color = (255,255,255)
            result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)
    
            # compute center offset
            x_center = (new_image_width - old_image_width) // 2
            y_center = (new_image_height - old_image_height) // 2
    
             # copy img image into center of result image
            result[y_center:y_center+old_image_height, 
            x_center:x_center+old_image_width] = crop_img
    
            char = cv2.resize(result,(224,224))
            char = char.astype("float") / 255.0
            char = img_to_array(char)
            char = np.expand_dims(char, axis=0)
            prob = model.predict(char)
            idx = np.argmax(prob)
            if height > width:
                cv2.putText(code_reg,labels[idx],(x-12,y+12),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),1)
            else:
                cv2.putText(code_reg,labels[idx],(x-5,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),1)
            cv2.rectangle(code_reg,(round(x),round(y)),(round(x+w),round(y+h)),(0,0,0),2)

    return code_reg