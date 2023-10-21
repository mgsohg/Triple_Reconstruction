import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Backbone import ConvAutoencoder
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

def Sec_REC():
    # Set random seed for reproducibility if needed
    torch.manual_seed(42)

    # Define data transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Define the paths for training and testing data
    train_data_path = '...'  # Replace with your training data path
    test_data_path = '...'   # Replace with your testing data path

    # Create datasets
    train_dataset = dsets.ImageFolder(train_data_path, transform=transform)
    test_dataset = dsets.ImageFolder(test_data_path, transform=transform)

    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is available() else "cpu")

    # Initialize the 2levCAE model
    model = ConvAutoencoder()
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Load a pre-trained model if needed
    load = None  # Replace with code to load a pre-trained model

    # TensorFlow SummaryWriter for logging
    log_dir = 'logs'  # Define your log directory
    writer = tf.summary.create_file_writer(log_dir)

    # Training parameters
    training_epochs = 20
    ToT_train_loss = []

    print('2lev CAE Start')
    print()

    for epoch in range(1, training_epochs + 1):
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
            train_output, _ = model(train_image)
            loss = criterion(train_output, train_image)

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
        graph(train_image)
        tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=log_dir)

if __name__ == "__main__":
    Sec_REC()
