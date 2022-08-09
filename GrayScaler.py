import cv2

class GrayScaler:
    def run(self, image): #Very basic gray scaler class
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)