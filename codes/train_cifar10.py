import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from model import CIFAR10Net
from vgg import VGG_A_BatchNorm, VGG_A
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torchsummary import summary
from torch.utils.data import random_split
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
import numpy as np
import torch.nn.functional as F

# Define Cutout transform
class Cutout(object):
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask

def get_dataloaders(batch_size=128, num_workers=2):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010)),
        Cutout(n_holes=1, length=16)
    ])

    # 新增一个没有增强的 transform
    train_eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=train_transform)
    trainset_eval = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=False, transform=train_eval_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    # 新增一个用于评估训练准确率的 loader
    train_eval_loader = torch.utils.data.DataLoader(trainset_eval, batch_size=100,
                                                    shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=num_workers)
    
    return trainloader, train_eval_loader, testloader

def visualize_filters(model, layer_num, num_filters=16):
    module = list(model.modules())[layer_num]
    if isinstance(module, nn.Conv2d):
        filters = module.weight.data.cpu().numpy()
        fig, axs = plt.subplots(4, 4, figsize=(10, 10))
        for i in range(num_filters):
            ax = axs[i//4, i%4]
            ax.imshow(filters[i, 0], cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"results/filters_layer_{layer_num}.png")
        plt.close()

def visualize_loss_landscape(model, criterion, inputs, targets, filename):
    model.eval()
    weights = []
    for param in model.parameters():
        if param.requires_grad:
            weights.append(param.data.clone())
    
    alphas = np.linspace(-1, 1, 20)
    betas = np.linspace(-1, 1, 20)
    losses = np.zeros((len(alphas), len(betas)))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            for param, w in zip(model.parameters(), weights):
                if param.requires_grad:
                    param.data = w + alpha * torch.randn_like(w) + beta * torch.randn_like(w)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses[i, j] = loss.item()
    
    for param, w in zip(model.parameters(), weights):
        if param.requires_grad:
            param.data = w

    plt.figure(figsize=(10, 8))
    plt.contourf(alphas, betas, losses.T, levels=50)
    plt.colorbar(label='Loss')
    plt.xlabel('α')
    plt.ylabel('β')
    plt.title('Loss Landscape')
    plt.savefig(filename)
    plt.close()



def evaluate_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

# def train_model(model, trainloader, train_eval_loader, testloader, device, model_name):
#     optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
#     criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

#     EPOCHS = 25
#     best_train_acc = 0.0
#     best_model_wts = None
#     loss_list, train_acc_list, test_acc_list = [], [], []

#     for epoch in range(EPOCHS):
#         model.train()
#         running_loss = 0.0
#         pbar = tqdm(trainloader, desc=f"{model_name} Epoch {epoch+1}/{EPOCHS}")

#         for inputs, labels in pbar:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#         epoch_loss = running_loss / len(trainloader)
#         loss_list.append(epoch_loss)

#         train_acc_epoch = evaluate_model(model, train_eval_loader, device)
#         test_acc_epoch = evaluate_model(model, testloader, device)
#         train_acc_list.append(train_acc_epoch)
#         test_acc_list.append(test_acc_epoch)

#         print(f"[{model_name} Epoch {epoch+1}] Train Loss: {epoch_loss:.4f}, "
#               f"Train Acc: {train_acc_epoch:.2f}%, Test Acc: {test_acc_epoch:.2f}%, "
#               f"LR: {optimizer.param_groups[0]['lr']:.6f}")

#         scheduler.step()

#         if train_acc_epoch > best_train_acc:
#             best_train_acc = train_acc_epoch
#             best_model_wts = model.state_dict()

#     print(f"{model_name} Best Training Accuracy: {best_train_acc:.2f}%")
#     return loss_list, train_acc_list, test_acc_list, best_model_wts

# def train():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     trainloader, train_eval_loader, testloader = get_dataloaders()

#     models = {
#         "VGG_A": VGG_A().to(device),
#         "VGG_A_BatchNorm": VGG_A_BatchNorm().to(device)
#     }

#     results = {}

#     for model_name, model in models.items():
#         summary(model, (3, 32, 32))
#         loss_list, train_acc_list, test_acc_list, best_model_wts = train_model(
#             model, trainloader, train_eval_loader, testloader, device, model_name
#         )
#         results[model_name] = {
#             "loss": loss_list,
#             "train_acc": train_acc_list,
#             "test_acc": test_acc_list,
#             "best_model_wts": best_model_wts
#         }

#         os.makedirs("results", exist_ok=True)
#         torch.save(best_model_wts, f"results/best_model_{model_name}.pth")

#     # Visualization
#     plt.figure(figsize=(12, 6))
#     for model_name in models:
#         plt.plot(results[model_name]["train_acc"], label=f"{model_name} Train Acc")
#         plt.plot(results[model_name]["test_acc"], label=f"{model_name} Test Acc")
#     plt.title("Training and Test Accuracy")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy (%)")
#     plt.legend()
#     plt.savefig("results/accuracy_comparison.png")
#     plt.close()

#     plt.figure(figsize=(12, 6))
#     for model_name in models:
#         plt.plot(results[model_name]["loss"], label=f"{model_name} Loss")
#     plt.title("Training Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.savefig("results/loss_comparison.png")
#     plt.close()

#     print("Training complete. Results saved to 'results/'")

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, train_eval_loader, testloader = get_dataloaders()
    model = CIFAR10Net().to(device)
    summary(model, (3, 32, 32))

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    EPOCHS = 25
    best_train_acc = 0.0
    best_model_wts = None
    loss_list, train_acc_list = [], []
    global_iter = 0

    vis_inputs, vis_targets = next(iter(trainloader))
    vis_inputs, vis_targets = vis_inputs.to(device), vis_targets.to(device)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if global_iter % 20 == 0:
                loss_list.append(running_loss / (i + 1))
            global_iter += 1

        train_acc_epoch = evaluate_model(model, train_eval_loader, device)
        print(f"[Epoch {epoch+1}] Train Loss: {running_loss/len(trainloader):.4f}, "
              f"Train Acc: {train_acc_epoch:.2f}%, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        train_acc_list.append(train_acc_epoch)

        scheduler.step()

        if train_acc_epoch > best_train_acc:
            best_train_acc = train_acc_epoch
            best_model_wts = model.state_dict()

        visualize_loss_landscape(model, criterion, vis_inputs, vis_targets, f"results/loss_landscape_epoch_{epoch+1}.png")

    print(f"Best Training Accuracy: {best_train_acc:.2f}%")
    os.makedirs("results", exist_ok=True)
    torch.save(best_model_wts, "results/best_model.pth")

    # Final Test Evaluation
    model.load_state_dict(best_model_wts)
    test_acc = evaluate_model(model, testloader, device)
    print(f"Test Accuracy (best train model): {test_acc:.2f}%")

    for i in range(3):
        visualize_filters(model, i + 1)

    # Visualization
    def plot_and_save(data, title, ylabel, filename, xlabel="Iteration"):
        plt.figure()
        plt.plot(data, label=title)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(f"results/{filename}")
        plt.close()

    plot_and_save(loss_list, "Training Loss", "Loss", "train_loss.png")
    plot_and_save(train_acc_list, "Training Accuracy", "Accuracy (%)", "train_accuracy.png")

    print("Training complete. Results saved to 'results/'")
 
if __name__ == '__main__':
    train()

