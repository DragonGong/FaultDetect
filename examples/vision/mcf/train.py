from vision_detect.data.dataloader import MultiCamDataset
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch.optim import Adam
from vision_detect.models.multi_camera_fusion import MultiCameraFusion
from tqdm import tqdm
import torch

dataset = MultiCamDataset('/Volumes/My Passport/dataset/kitti/data3', cam_list=['front', 'left', 'right']
                          )
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = MultiCameraFusion(load_pretrained=True,
                          base_model_path='/Volumes/My Passport/dataset/models/pretrained/resnet18-multi.pth'
                          , freeze_pretrained=True)
loss = nn.CrossEntropyLoss()
opt = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('mps')
model.to(device)


def loss_function(outputs, labels):
    l = 0
    for i in range(outputs.size(1)):
        # output B  N  2  label B N   ->  [B,2] and [B]
        l += loss(outputs[:, i, :], labels[:, i])
    l /= outputs.size(1)
    return l


def pred_function(outputs, labels):
    _, pred = outputs.max(2)
    correct = 0
    for i, b_pred in enumerate(pred):
        correct_batch = b_pred.eq(labels[i]).sum(0).item()
        correct += correct_batch / labels.size(1)
    return correct


def train_model(model, train_loader, val_loader, loss_func, opt, pred_func, device=torch.device('cpu'), nums_epoch=50):
    model.train()
    best_val_loss = float('inf')
    best_model = None

    for epoch in tqdm(range(nums_epoch)):
        train_loss = 0.0
        with tqdm(train_loader, desc='train') as t:
            for images, labels in t:
                images, labels = images.to(device), labels.to(device)
                opt.zero_grad()
                outputs = model(images)
                running_loss = loss_func(outputs, labels)
                running_loss.backward()
                train_loss += running_loss.item() * images.size(0)
                opt.step()
                t.set_postfix(loss=running_loss.item())

        train_loss = train_loss / len(train_loader.dataset)
        model.eval()

        val_loss = 0.0
        correct = 0
        with tqdm(val_loader, desc='val') as t:
            for images, labels in t:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                running_loss = loss_func(outputs, labels)
                val_loss += running_loss.item() * images.size(0)
                _, pred = outputs.max(1)
                correct += pred_func(outputs, labels)
                t.set_postfix(loss=running_loss.item())
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100. * correct / len(val_loader.dataset)
        if val_loss < best_val_loss:
            best_model = model.state_dict().copy()
            best_val_loss = val_loss
            torch.save(best_model, '/Volumes/My Passport/dataset/models/trained/8_5/best.pth')
            print(f'Saved best model at epoch {epoch + 1}')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    model.load_state_dict(best_model)
    return model


if __name__ == '__main__':
    model = train_model(model, train_loader, val_loader, loss_function, opt, pred_function, device, 30)
    torch.save(model.state_dict(), '/Volumes/My Passport/dataset/models/trained/8_5/final.pth')
