from __future__ import division, print_function
import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torchvision import transforms
import numpy as np
import argparse

class ConvAutoencoder(nn.Module):
    # Your ConvAutoencoder class definition here

def feature_extraction(data_loader, load, dual, output_dir, device):
    load.to(device)
    dual.to(device)

    print('Start Feature Extraction')
    print()
    for epoch in range(1, 2):
        load.eval()
        Feature_Y = np.empty([0, 1])
        Feature_X = []
        i = 0
        for data1 in data_loader:
            images, label = data1

            labels_np = label.numpy()
            Feature_Y = np.concatenate((Feature_Y, labels_np), axis=None)

            # final
            images = images.to(device)
            output__, features__ = load(images)
            output__ = output__.to(device)
            output, features = dual(output__)

            features = features.cpu().detach().numpy()
            features = features.flatten()
            Feature_X.append(features)

        Feature_X = np.array(Feature_X)
        # Save Feature_X and Feature_Y separately (This is for additional networks)

def main(args):
    # Define data transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Create DataLoader for original data
    Loader = dsets.ImageFolder(args.data_path, transform=transform)
    Ext_Loader = torch.utils.data.DataLoader(dataset=Loader, batch_size=1, shuffle=False)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pretrained models (replace None with actual model loading)
    load_baseline = None
    load_additional = None
    dual = None

    if args.baseline:
        feature_extraction(Ext_Loader, load_baseline, dual, args.output_dir, device)

    if args.additional:
        feature_extraction(Ext_Loader, load_additional, dual, args.output_dir, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Extraction Script")
    parser.add_argument('--data_path', type=str, help='Path to data folder', required=True)
    parser.add_argument('--output_dir', type=str, help='Output directory for features', required=True)
    parser.add_argument('--baseline', action='store_true', help='Perform baseline feature extraction')
    parser.add_argument('--additional', action='store_true', help='Perform additional feature extraction')

    args = parser.parse_args()
    main(args)
