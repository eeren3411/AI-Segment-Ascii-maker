import cv2
import mediapipe as mp
import numpy as np
import torch
from detectron2.engine.defaults import DefaultPredictor
from adet.config import get_cfg

#media pipe selfie segmentation implementation
#You can read more about it in 
#https://google.github.io/mediapipe/solutions/selfie_segmentation.html
class HumanSegmentator: 
    def __init__(self):
        self.selfieSegmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
        
    def run(self, image):
        RGBimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.selfieSegmentation.process(RGBimage)

        condition = np.stack((result.segmentation_mask,) * 3, axis=-1) > 0.1
        background = np.zeros(image.shape, dtype=np.uint8)
        background[:] = (0, 0, 0)
        output = np.where(condition, image, background)
        return output

#Yet another anime segmentator by zymk9 and koke2c95
#You can read more about it in
#https://github.com/zymk9/Yet-Another-Anime-Segmenter

#Here is a great demo to show how it works by hysts
#You can find it here
#https://huggingface.co/spaces/hysts/Yet-Another-Anime-Segmenter

#I have literally no idea how detectron or adelai functions work. I just learnt how to use them from hysts' code.
class AnimeSegmentator:
    def __init__(self):
        config_path = "./YAAS/SOLOv2.yaml"
        model_path = "./YAAS/SOLOv2.pth"
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.freeze()
        self.model = DefaultPredictor(cfg)
        self.model.score_threshold = 0.1
        self.model.mask_threshold = 0.5
    
    def run(self, image):
        preds = self.model(image)
        mask = preds['instances'].pred_masks.cpu().numpy().astype(int).max(axis=0)
        output = image.copy()[:, :, ::-1]
        output[mask == 0] = 0
        return output