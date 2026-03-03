import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from collections import OrderedDict

def get_input_args():
    ''' command-line arguments for training '''
    parser = argparse.ArgumentParser(description='Train a neural network for image classification')
    parser.add_argument('data_dir', type=str, help='Path to the dataset directory')
    parser.add_argument('--save_dir', type=str, default='./', help='Directory to save the checkpoint')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg13', 'vgg16'], help='Pretrained architecture to use')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    return parser.parse_args()

def load_data(data_dir):
    ''' Load and preprocess the data '''
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=True)

    return train_loader, valid_loader, train_data

def build_model(arch, hidden_units):
    ''' Build the model based on the chosen architecture '''
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False  # Freeze the pretrained layers

    # Classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units, len(train_data.classes))),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    return model

def train_model(model, train_loader, valid_loader, learning_rate, epochs, gpu):
    ''' Train the model '''
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}.. Train loss: {running_loss/len(train_loader):.3f}")
        
        # Validation loss and accuracy
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                logps = model(images)
                loss = criterion(logps, labels)
                val_loss += loss.item()

                ps = torch.exp(logps)
                top_class = ps.argmax(dim=1)
                val_accuracy += (top_class == labels).float().mean().item()

        print(f"Validation loss: {val_loss/len(valid_loader):.3f}.. Validation accuracy: {val_accuracy/len(valid_loader)*100:.2f}%")

    return model

def save_checkpoint(model, save_dir, arch, hidden_units, learning_rate, epochs, gpu):
    ''' Save the trained model as a checkpoint '''
    checkpoint = {
        'arch': arch,
        'state_dict': model.state_dict(),
        'class_to_idx': train_data.class_to_idx,
        'classifier': model.classifier,
        'epochs': epochs,
        'learning_rate': learning_rate
    }
    torch.save(checkpoint, save_dir + '/checkpoint.pth')
    print(f"Checkpoint saved to {save_dir}/checkpoint.pth")

def main():
    args = get_input_args()
    train_loader, valid_loader, train_data = load_data(args.data_dir)
    model = build_model(args.arch, args.hidden_units)
    model = train_model(model, train_loader, valid_loader, args.learning_rate, args.epochs, args.gpu)
    save_checkpoint(model, args.save_dir, args.arch, args.hidden_units, args.learning_rate, args.epochs, args.gpu)

if __name__ == "__main__":
    main()
