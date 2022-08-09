import cv2
import numpy as np

class AsciiMaker:
    def __init__(self, segmentator="Human", grayScaler=None, targetWidth = 150):
        if segmentator == "Normal" or segmentator == "Human":
            from Segmentator import HumanSegmentator
            self.segmentator = HumanSegmentator()
        elif segmentator == "Anime":
            from Segmentator import AnimeSegmentator
            self.segmentator = AnimeSegmentator()
        else:
            self.segmentator = segmentator #You can also make your own segmentator. It just needs to segment object and return image with black background
        if grayScaler == None:
            from GrayScaler import GrayScaler
            self.grayScaler = GrayScaler()
        else:
            self.grayScaler = grayScaler #you can also put your own grayscaler. I'm using cv2's grayscaler for this in default

        #these are the characters program uses according to value in grayscale
        self._25 = "." 
        self._50 = "-"
        self._75 = "+"
        self._100 = "?"
        self._125 = "£"
        self._150 = "X"
        self._175 = "$"
        self._200 = "&"
        self._225 = "#"
        self._else = "@"

        #target width in characters
        self._targetWidth = targetWidth

    def run(self, image):

        if(type(image) == type("string")): #If input is a string, program assumes that its path to image and reads it as image
            image = cv2.imread(image)

        if(self.segmentator != None): #If segmentation isn't disabled, segment image
            #Segment Image
            image = self.segmentator.run(image)

        #Resize Image
        iHeight, iWidth, _ = image.shape
        scaleFactor = self._targetWidth/iWidth
        targetHeight = int(scaleFactor*iHeight/2) #since characters in terminal like 8 pixels wide and 15 pixels tall. I'm dividing height by 2 to maintain aspect ratio to some extend
        targetDims = (self._targetWidth, targetHeight)

        resizedImg = cv2.resize(image, targetDims, interpolation=cv2.INTER_AREA)

        #Grayscale Image
        grayImg = self.grayScaler.run(resizedImg)

        #Dark images are usually pretty hard to convert
        #so I'm just multiplying the value by a scale factor
        gammaScaler = 1
        if(self.segmentator == None):
            gammaScaler = 100/np.average(grayImg)
        else:
            gammaScaler = 35/np.average(grayImg)
        
        gammaScaler = max(1, gammaScaler)

        for y in range(targetDims[1]):
            for x in range(targetDims[0]):
                value = grayImg[y][x] * gammaScaler
                if value < 25:
                    print(self._25, end="")
                elif value < 50:
                    print(self._50, end="")
                elif value < 75:
                    print(self._75, end="")
                elif value < 100:
                    print(self._100, end="")
                elif value < 125:
                    print(self._125, end="")
                elif value < 150:
                    print(self._150, end="")
                elif value < 175:
                    print(self._175, end="")
                elif value < 200:
                    print(self._200, end="")
                elif value < 225:
                    print(self._225, end="")
                else:
                    print(self._else, end="")
            print()

def main():
    asciiMaker = AsciiMaker(segmentator="Anime")
    asciiMaker.run("./hutao.png")

def test():
    print(".") #25
    print("-") #50
    print("+") #75
    print("?") #100
    print("£") #125
    print("X") #150
    print("$") #175
    print("&") #200
    print("#") #225
    print("@") #else

if __name__ == "__main__":
    main()