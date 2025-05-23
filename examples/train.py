from data.dataloader import OcclusionDetectionDataset
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch.optim import Adam
from models.OcclusionDetectionModel import OcclusionDetectionModel
from tqdm import tqdm
import torch

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)
dataset = OcclusionDetectionDataset(data_dir='/Volumes/My Passport/dataset/kitti/data/train',
                                    label_file='/Volumes/My Passport/dataset/kitti/data/train/labels.txt',
                                    transform = transform
                                    )
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = OcclusionDetectionModel(base_model_path='/Volumes/My Passport/dataset/models/pretrained/resnet18-f37072fd.pth'
                                , freeze_pretrained=True)
loss = nn.CrossEntropyLoss()
opt = Adam(filter(lambda p:p.requires_grad ,  model.parameters()), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def train_model(model, train_loader, val_loader, loss_func, opt, device=torch.device('cpu'), nums_epoch=50):
    model.train()
    best_val_loss = float('inf')
    best_model = None

    for epoch in tqdm(range(nums_epoch)):
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            outputs = model(images)
            running_loss = loss_func(outputs, labels)
            running_loss.backward()
            train_loss += running_loss.item() * images.size(0)
            opt.step()

        train_loss = train_loss / len(train_loader.dataset)
        model.eval()

        val_loss = 0.0
        correct = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            running_loss = loss_func(outputs, labels)
            val_loss += running_loss.item() * images.size(0)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum(0).item()
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100. * correct / len(val_loader.dataset)
        if val_loss < best_val_loss:
            best_model = model.state_dict().copy()
            best_val_loss = val_loss
            print(f'Saved best model at epoch {epoch + 1}')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    model.load_state_dict(best_model)
    return model


if __name__ == '__main__':
    train_model(model, train_loader, val_loader, loss, opt, device, 100)
