# packaging the train module to a class and test it there

from vision_detect.data.dataloader import MultiCamDataset
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch.optim import Adam
from vision_detect.models.multi_camera_fusion import MultiCameraFusion
from vision_detect.train.trainer import Trainer
import torch

dataset = MultiCamDataset(
    '/Volumes/My Passport/dataset/kitti/data3',
    cam_list=['front', 'left', 'right']
)

train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = MultiCameraFusion(
    load_pretrained=True,
    base_model_path='/Volumes/My Passport/dataset/models/pretrained/resnet18-multi.pth',
    freeze_pretrained=True
)

loss = nn.CrossEntropyLoss()
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)


def loss_function(outputs, labels):
    local_cls = outputs[0]
    global_cls = outputs[1]
    l = 0
    for i in range(local_cls.size(1)):
        l += loss(local_cls[:, i, :], labels[:, i])
    local_loss = l / local_cls.size(1)
    l = 0
    for i in range(global_cls.size(1)):
        g_label = labels.max(dim=1, keepdim=True).values
        l += loss(global_cls[:, i, :], g_label[:, i])
    global_loss = l / global_cls.size(1)
    return local_loss * 1 + global_loss * 0.5


def pred_function(outputs, labels):
    pred_output = outputs[0]
    _, pred = pred_output.max(dim=2)
    correct = 0
    for i, b_pred in enumerate(pred):
        correct_batch = b_pred.eq(labels[i]).sum(0).item()
        correct += correct_batch / labels.size(1)
    return correct


device = torch.device('mps')
model.to(device)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_function,
    optimizer=optimizer,
    pred_fn=pred_function,
    device=device,
    save_dir='assets/mcf-model/test'
)

if __name__ == '__main__':
    trained_model = trainer.train(num_epochs=30)
    print("Training completed.")
