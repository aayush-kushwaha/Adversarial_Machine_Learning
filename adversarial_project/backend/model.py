import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CIFARModelWrapper:
    def __init__(self, model_path):
        self.device = device
        self.model = SimpleCNN().to(self.device)
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {self.model_path}. Place model_cifar.pt in saved_model/."
            )
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.transform = transforms.ToTensor()

    def preprocess(self, image):
        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)

    def predict_logits(self, x):
        with torch.no_grad():
            return self.model(x.to(self.device))

    def predict_class(self, x):
        logits = self.predict_logits(x)
        return logits.argmax(dim=1).item()


def get_model(model_path=None):
    base_path = Path(__file__).resolve().parents[1]
    if model_path is None:
        model_path = base_path / "saved_model" / "model_cifar.pt"
    return CIFARModelWrapper(model_path)
