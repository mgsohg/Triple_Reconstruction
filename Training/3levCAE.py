import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Backbone import ConvAutoencoder
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse

def Third_REC(data_path, log_dir, pretrain_model_path, epochs=20, batch_size=64):
    # Set random seed for reproducibility if needed
    torch.manual_seed(42)

    # Define data transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Create datasets
    train_dataset = dsets.ImageFolder(data_path, transform=transform)

    # Create data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the 3levCAE model
    model = ConvAutoencoder()
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Load a pre-trained model if needed
    load = None  # Replace with code to load a pre-trained model

    # TensorFlow SummaryWriter for logging
    writer = tf.summary.create_file_writer(log_dir)

    # Training parameters
    ToT_train_loss = []

    print('3lev CAE Start')
    print()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for i, data in enumerate(train_loader):
            train_image, train_label = data
            train_image = train_image.to(device)

            if load is not None:
                output_rec, features = load(train_image)
                output_rec = output_rec.to(device)
            else:
                features = None  # Modify this as needed for your use case

            optimizer.zero_grad()
            train_output, _ = model(output_rec)
            loss = criterion(train_output, train_image) # Third Target Mapping

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * train_image.size(0)

        # Calculate average training loss for the epoch
        train_loss = train_loss / len(train_loader)
        ToT_train_loss.append(train_loss)

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

        # Log training loss to TensorFlow
        with writer.as_default():
            tf.summary.scalar('Training Loss', train_loss, step=epoch)

    # Save the model's computational graph to TensorFlow log directory
    with writer.as_default():
        graph = tf.function(model.forward)  # Convert PyTorch model to TensorFlow function
        tf.summary.trace_on()
        graph(output_rec)  # Use output_rec for the graph, replace as needed
        tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=log_dir)

    # Additional code for saving the third model can be added here

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a 3levCAE model.")
    parser.add_argument('--data_path', type=str, help='Path to training data', required=True)
    parser.add_argument('--log_dir', type=str, help='Directory for TensorFlow logs', required=True)
    parser.add_argument('--pretrain_model_path', type=str, help='Path to a pre-trained model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')

    args = parser.parse_args()
    Third_REC(args.data_path, args.log_dir, args.pretrain_model_path, args.epochs, args.batch_size)
