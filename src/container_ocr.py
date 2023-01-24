import cv2
import argparse
from detect_bounding_box import detect_code
from ocr import ocr


def container_ocr():
    parse = argparse.ArgumentParser()
    parse.add_argument("-i","--image")
    args = vars(parse.parse_args())

    src_img = cv2.imread(args['image'])
    
    cv2.imshow("input",src_img)
    cv2.waitKey()
    cv2.destroyWindow('input')
    
    
    code_reg = detect_code(src_img)
    cv2.imshow('code_region',code_reg)
    cv2.waitKey()
    cv2.destroyWindow('code_region')

    result = ocr(code_reg)
    cv2.imshow('result',result)
    cv2.waitKey()

def main():
     container_ocr()

if __name__ == "__main__":
    main()
