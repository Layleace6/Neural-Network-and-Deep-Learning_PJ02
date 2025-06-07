import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from vgg import VGG_A
from vgg import VGG_A_BatchNorm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.datasets as datasets

class PartialDataset(Dataset):
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self):
        return self.dataset.__getitem__()

    def __len__(self):
        return min(self.n_items, len(self.dataset))


def get_cifar_loader(root='./data', batch_size=128, train=True, shuffle=True, num_workers=4, n_items=-1):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))

    data_transforms = transforms.Compose(
        [transforms.ToTensor(),
        normalize])

    dataset = datasets.CIFAR10(root=root, train=train, download=False, transform=data_transforms)
    if n_items > 0:
        dataset = PartialDataset(dataset, n_items)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)
for X, y in train_loader:
    img = np.transpose(X[0].cpu().numpy(), (1, 2, 0))
    break

def get_accuracy(model, data_loader, is_train=True):
    if is_train == False:
        model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            x, y = data
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = correct / total
    model.train() 
    return accuracy

def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate(model, dataloader, criterion):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def train(model, optimizer, criterion, train_loader, val_loader, test_loader=None, scheduler=None, epochs_n=30, duration=30):
    model.to(device)
    loss_values = []
    grad_distances = []
    beta_values = []

    iteration = 0
    accumulated_loss = 0
    previous_grad = None
    previous_param = None

    for epoch in range(epochs_n):
        model.train()
        if scheduler is not None:
            scheduler.step()

        epoch_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs_n}", ncols=80)
        for data in pbar:
            iteration += 1
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate loss and accuracy for this epoch
            epoch_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Every `duration` steps, record loss, gradient distance, beta smoothness
            accumulated_loss += loss.item()
            if iteration % duration == 0:
                current_grad = model.classifier[-1].weight.grad.detach().clone()
                current_param = model.classifier[-1].weight.detach().clone()

                if previous_grad is not None:
                    grad_distance = torch.dist(current_grad, previous_grad).item()
                    grad_distances.append(grad_distance)

                if previous_param is not None:
                    param_distance = torch.dist(current_param, previous_param).item()
                    beta_values.append(grad_distance / (param_distance + 1e-3))

                previous_grad = current_grad
                previous_param = current_param
                loss_values.append(accumulated_loss / duration)
                accumulated_loss = 0

        # Compute and print training accuracy and loss
        avg_loss = epoch_loss / total
        accuracy = correct / total
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")

        # Optional: evaluate on validation set each epoch
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion)
            print(f"           Val Loss = {val_loss:.4f}, Accuracy = {val_acc:.4f}")

    # Final test accuracy
    if test_loader is not None:
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(f"Final Test Accuracy: {test_acc:.4f}")

    return loss_values, grad_distances, beta_values


def plot_loss_landscape(ax, list_vgg, list_vgg_bn, title, ylabel, label_vgg, label_vgg_bn, duration):
    min_vgg = np.min(list_vgg, axis=0)
    max_vgg = np.max(list_vgg, axis=0)
    steps = np.arange(0, len(min_vgg)) * duration
    ax.plot(steps, min_vgg, 'g-', alpha=0.8, label=label_vgg)
    ax.plot(steps, max_vgg, 'g-', alpha=0.8)
    ax.fill_between(steps, min_vgg, max_vgg, color='g', alpha=0.4)
    
    min_vgg_bn = np.min(list_vgg_bn, axis=0)
    max_vgg_bn = np.max(list_vgg_bn, axis=0)
    steps = np.arange(0, len(min_vgg_bn)) * duration
    ax.plot(steps, min_vgg_bn, 'r', alpha=0.8, label=label_vgg_bn)
    ax.plot(steps, max_vgg_bn, 'r', alpha=0.8)
    ax.fill_between(steps, min_vgg_bn, max_vgg_bn, color='r', alpha=0.4)
    
    ax.set(title=title, ylabel=ylabel, xlabel='Iterations')
    ax.legend()


def plot_gradient_distance(ax, list_vgg, list_vgg_bn, title, ylabel, label_vgg, label_vgg_bn, duration):
    min_vgg = np.min(list_vgg, axis=0)
    max_vgg = np.max(list_vgg, axis=0)
    steps = np.arange(0, len(min_vgg)) * duration
    ax.plot(steps, min_vgg, 'g-', alpha=0.8, label=label_vgg)
    ax.plot(steps, max_vgg, 'g-', alpha=0.8)
    ax.fill_between(steps, min_vgg, max_vgg, color='g', alpha=0.4)
    
    min_vgg_bn = np.min(list_vgg_bn, axis=0)
    max_vgg_bn = np.max(list_vgg_bn, axis=0)
    steps = np.arange(0, len(min_vgg_bn)) * duration
    ax.plot(steps, min_vgg_bn, 'r', alpha=0.8, label=label_vgg_bn)
    ax.plot(steps, max_vgg_bn, 'r', alpha=0.8)
    ax.fill_between(steps, min_vgg_bn, max_vgg_bn, color='r', alpha=0.4)
    
    ax.set(title=title, ylabel=ylabel, xlabel='Iterations')
    ax.legend()
    
def plot_beta_smoothness(ax, list_vgg, list_vgg_bn, title, ylabel, label_vgg, label_vgg_bn, duration):
    max_vgg = np.max(np.asarray(list_vgg), axis=0)
    max_vgg_bn = np.max(np.asarray(list_vgg_bn), axis=0)
    steps = np.arange(0, len(max_vgg)) * duration
    ax.plot(steps, max_vgg, 'g-', alpha=0.8, label=label_vgg)
    ax.plot(steps, max_vgg_bn, 'r', alpha=0.8, label=label_vgg_bn)
    
    ax.set(title=title, ylabel=ylabel, xlabel='Iterations')
    ax.legend()


if __name__ == '__main__':
    epochs = 30
    learning_rates = [0.15, 0.1, 0.075, 0.05]
    duration = 30  
    set_random_seeds(seed_value=2020, device=device)
    
    grad_list_vgg = []
    loss_list_vgg = []
    beta_list_vgg = []
    grad_list_vgg_bn = []
    loss_list_vgg_bn = []
    beta_list_vgg_bn = []
    
    for lr in learning_rates:
        print(f'Training Standard VGG-A, learning rate: {lr}')
        model_vgg = VGG_A()
        optimizer = torch.optim.SGD(model_vgg.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        loss_vals, grads, beta_vals = train(model_vgg, optimizer, criterion, train_loader, val_loader, epochs_n=epochs, duration=duration)
        grad_list_vgg.append(grads)
        loss_list_vgg.append(loss_vals)
        beta_list_vgg.append(beta_vals)
    
    for lr in learning_rates:
        print(f'Training VGG-A with BatchNorm, learning rate: {lr}')
        model_vgg_bn = VGG_A_BatchNorm()
        optimizer = torch.optim.SGD(model_vgg_bn.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        loss_vals, grads, beta_vals = train(model_vgg_bn, optimizer, criterion, train_loader, val_loader, epochs_n=epochs, duration=duration)
        grad_list_vgg_bn.append(grads)
        loss_list_vgg_bn.append(loss_vals)
        beta_list_vgg_bn.append(beta_vals)
    
    
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(9, 6), dpi=800)
    plot_loss_landscape(ax, np.array(loss_list_vgg), np.array(loss_list_vgg_bn), 'Loss Landscape', 'Loss', 'Standard VGG', 'VGG with BatchNorm', duration)
    plt.savefig('results/Loss_Landscape.png')
    plt.close()
    
    # Plot Gradient Distance
    fig, ax = plt.subplots(figsize=(9, 6), dpi=800)
    plot_gradient_distance(ax, np.array(grad_list_vgg), np.array(grad_list_vgg_bn), 'Gradient Predictiveness', 'Gradient Distance', 'Standard VGG', 'VGG with BatchNorm', duration)
    plt.savefig('results/Gradient_Distance.png')
    plt.close()
    
    # Plot Beta Smoothness
    fig, ax = plt.subplots(figsize=(9, 6), dpi=800)
    plot_beta_smoothness(ax, np.array(beta_list_vgg), np.array(beta_list_vgg_bn), 'Beta Smoothness', 'Beta', 'Standard VGG', 'VGG with BatchNorm', duration)
    plt.savefig('results/Beta_Smoothness.png')
    plt.close()
