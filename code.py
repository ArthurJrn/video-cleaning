# Code complet 

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from torch import nn

# Hyperparmètres
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Fonctions auxiliaires
def predict_class(filename, model):
    """ Fonction prédisant la classe d'une image par un modèle.
	
	Args: 
		- filename: str, path de l'image
		- model: nn.Sequential, PyTorch modele
	Returns:
		- index: int, index de la valeur max en sortie du réseau
    """
    img = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    img_processed = preprocess(img)
    batch_img = torch.unsqueeze(img_processed, 0)
    out = model(batch_img)
    _, index = torch.max(out, 1)

    return index
    
def cosine_similarity(vector_1, vector_2):
    """ Fonction calculant la similarité cosinus entre deux vecteurs.
	Args:
		- vector_1: torch.Tensor
		- vector_2: torch.Tensor
	Returns:
		sim: float, similarité cosinus entre les deux vecteurs.
    """
    sim = torch.dot(vector_1, vector_2) / (np.linalg.norm(vector_1)*np.linalg.norm(vector_2))
    return sim

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

def extract_features(img_path, model):
    """ Fonction extrayant les features d'une images a partir d'un modèle.

    Agrs:
        - img_path: str, path de l'image
        - model: nn.Sequential, modele performant la feature extraction
    
    Returns: features, torch.Tensor de shape (1, 2048)
    """

    img = Image.open(img_path)
    img_processed = preprocess(img)
    batch_image = torch.unsqueeze(img_processed, 0)

    with torch.no_grad():
        features = torch.flatten(feature_extractor(batch_image))
    return features

def get_best_similarity(frame_features, paths):
    """ Fonction renvoyant le path de l'image la plus similaire a une frame donnée.

    Args:
        - frame_features: torch.Tensor de shape (1, 2048), features de la frame
        - paths: list, liste de paths 
    
    Returns:
        - best_image: str, path de la frame la plus similaire à la frame en entrée.
    """
    
    best_sim = 0
    best_image = ""
    
    for img_path in paths:
        sim = cosine_similarity(video_features[img_path], frame_features)
        
        if sim > best_sim:
            best_sim = sim
            best_image = img_path
    return best_image
    
# Chargement de la vidéo
video_name = "shuffled_video_360.mp4"
video_path = os.path.abspath(video_name)

# Open the video
video = cv2.VideoCapture(video_path)

# Get number of frames
nb_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = video.get(cv2.CAP_PROP_FPS)
print(f'La vidéo est constituée de {nb_frames} images, avec {fps} fps. Format: {width, height}')


# Chargement du modèle pré-entraîné
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=True)
model.eval()

# Chargement des classes ImageNet
with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]
    

# Approche 1
paths = [os.path.join(os.path.abspath("frames"), filename) for filename in os.listdir("frames")]

# Dictionnaire: les clés sont les classes, les valeurs sont les listes de frames appartenant à cette classe
classes_by_frames = {}

for filename in paths:

    index = predict_class(filename, model=model)
    predicted_class = classes[index[0]]

    if predicted_class in classes_by_frames:
        classes_by_frames[predicted_class].append(filename)
    else:
        classes_by_frames[predicted_class] = [filename]

# Un peu de cleaning
classes_by_frames['alp'] = classes_by_frames["970: 'alp',"]
del classes_by_frames["970: 'alp',"]

# On enlève les frames n'appartenant pas à la classe alp
bad_frames = []
for filename in paths:
    if filename not in classes_by_frames['alp']:
        bad_frames.append(filename)

print(f"Avec la première approche, on enlève {len(bad_frames)} frames.")

# Approche 2
feature_extractor = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=True)
feature_extractor.fc = Identity()
feature_extractor.eval()

# Note: j'ai ici combiné les deux méthodes en enlevant les frames éliminées par la méthodes 1
paths = [os.path.join(os.path.abspath("frames"), filename) for filename in os.listdir("frames") if filename not in bad_frames]

# Dict: les clés sont les paths des images, les valeurs sont les features associées
video_features = {img_path:extract_features(img_path, model=feature_extractor) for img_path in paths}

# On part de la frame n°1 et on trouve itérativement la frame suivante par similarité
current_frame = os.path.abspath("frames/frame_no_1.png")
ordered_video = [current_frame]
paths = [os.path.join(os.path.abspath("frames"), filename) for filename in os.listdir("frames") if filename not in bad_frames and filename !="frame_no_1.png"]
while paths:

    # Calcul de la similarité entre current_image et les autres frames
    best_next_image = get_best_similarity(video_features[current_frame], paths)
    ordered_video.append(best_next_image)

    # Une image ne pouvant être qu'une fois dans la vidéo, on enlève l'image sélectionée
    paths.remove(best_next_image)
    current_frame = best_next_image

# Ecriture de la vidéo
frameSize = (640, 360)

out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, frameSize)
for img_path in ordered_video:
    img = cv2.imread(img_path)
    #print(img)
    #gray = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    out.write(img)

out.release()