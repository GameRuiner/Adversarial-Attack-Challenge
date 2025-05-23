""""Module to extract template from the deep model"""
import torch
import numpy as np
import cv2
from model import SiameseNetwork
import albumentations as A
from albumentations.pytorch import ToTensorV2
from face_utils import detect_face_haar 

class TemplateExtractor:
    """"Class to extract template data from face images"""

    def __init__(self, model_path="fr_pgd3.pth"):
        """
        Implement custom initialization here
        """
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = SiameseNetwork().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = A.Compose([
            A.Resize(112, 112),
            A.Normalize(),
            ToTensorV2()
        ])


    @staticmethod
    def compare(reference, probe):
        """
        Compares two already L2-normalized vectors (as returned by the model).
        Returns cosine similarity in range [-1, 1].
        """
        return float(np.dot(reference.flatten(), probe.flatten()))


    def extract(self, image_path):
        """
            Method to extract feature template from image path. Implement custom logic here.
            Input - str path to the image.
            Returns numpy array.
        """
       
        #read image and extract template
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at path: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = detect_face_haar(img)
        if self.transform:
            img_t = self.transform(image=img)['image']
        img_t = img_t.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.forward_once(img_t)
        return embedding.squeeze(0).cpu().numpy()