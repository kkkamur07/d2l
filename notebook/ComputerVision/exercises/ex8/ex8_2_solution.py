import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# --- Config ---
batch_size = 256
epochs = 10
temperature = 0.5
embedding_dim = 128
lr = 1e-3

# --- Data Augmentations ---


class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.transforms = T.Compose(
            [
                T.RandomResizedCrop(32, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.4, 0.4, 0.4, 0.1),
                T.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        x, _ = self.base_dataset[index]
        xi = self.transforms(x)
        xj = self.transforms(x)
        return xi, xj

    def __len__(self):
        return len(self.base_dataset)


# --- Simple Modern ConvNet Encoder ---
class SimpleConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.net(x)
        return x.view(x.size(0), -1)


# --- Encoder and Projection Head ---
class SimCLRModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.encoder = resnet18(pretrained=False, num_classes=4)  # output ignored
        # self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # CIFAR-10 input size
        # self.encoder.maxpool = nn.Identity()  # remove maxpool layer
        # self.encoder.fc = nn.Identity()  # remove final fc layer
        self.encoder = SimpleConvEncoder()
        self.projector = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, embedding_dim))

    def forward(self, x):
        h = self.encoder(x).squeeze()
        z = self.projector(h)
        return F.normalize(z, dim=1)


# --- Contrastive Loss (NT-Xent) ---
def contrastive_loss(z1, z2, temperature=0.5):
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2N, D]
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # [2N, 2N]

    sim /= temperature

    # mask out self-similarities
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)

    # construct positive indices
    positives = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)

    loss = F.cross_entropy(sim, positives)
    return loss


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device "{device}".')

    model = SimCLRModel()
    if not os.path.exists("simclr_model.pth"):
        print("Model file simclr_model.pth does not exist. Starting training from scratch.")
    else:
        print("Model file simclr_model.pth found. Loading existing model.")
        model.load_state_dict(torch.load("simclr_model.pth", map_location="cpu"))

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_dataset = torchvision.datasets.CIFAR10(root="data", train=True, download=False)
    train_loader = DataLoader(SimCLRDataset(train_dataset), batch_size=batch_size, shuffle=True, num_workers=1)

    # --- Training Loop ---
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xi, xj in tqdm(train_loader):
            xi, xj = xi.to(device), xj.to(device)
            zi, zj = model(xi), model(xj)
            loss = contrastive_loss(zi, zj, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

        # Save the model
        print("Saving model...")
        torch.save(model.state_dict(), "simclr_model.pth")
        print("Model saved as simclr_model.pth")
