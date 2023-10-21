from __future__ import division, print_function
import argparse
import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torchvision import transforms

# Import your ConvAutoencoder from Backbone

def train_baseline_model(data_path, save_path, epochs=40, batch_size=64):
    # Define data transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Create training dataset
    train_dataset = dsets.ImageFolder(data_path, transform=transform)

    # Create data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the baseline network model
    model = ConvAutoencoder()
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print('Pretraining Start')
    print()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for data in train_loader:
            train_image, train_label = data
            train_image = train_image.to(device)
            optimizer.zero_grad()
            train_output, _ = model(train_image)

             ## GENERIC TRAINING SCHEME
            loss = criterion(train_output, train_image)  

            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * train_image.size(0)

        # Calculate and print average training statistics for the epoch
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    # Save the trained model
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a ConvAutoencoder model.")
    parser.add_argument('--data_path', type=str, help='Path to training data', required=True)
    parser.add_argument('--save_path', type=str, help='Path to save the trained model', required=True)
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')

    args = parser.parse_args()
    train_baseline_model(args.data_path, args.save_path, args.epochs, args.batch_size)
