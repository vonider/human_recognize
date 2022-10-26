from torchvision.models.mobilenet import mobilenet_v2
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
import albumentations as A
import torch
import cv2
import sys

device = torch.device("cpu")
transform = A.Compose(
    [
    A.Resize(256, 256),
    A.Normalize((0.1307,), (0.3081,)),
    ToTensorV2(),
    ], p=1.0)

image_path = sys.argv[1]

image = cv2.imread(image_path)
aug = transform(image=image)
image = aug['image']

model = mobilenet_v2()
model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=2)
model.to(device)
model.load_state_dict(torch.load('trained_model.pt', map_location=device))
model.eval()
img = image.to(device)
img = img.unsqueeze(0)
output = model(img)
predictions = output.softmax(dim=1)[0][1].item()
print('Probability that there is a person in the image: {:.2%}'.format(predictions))