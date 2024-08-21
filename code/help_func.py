import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reconstruction error : mean + 3 * standard deviation
def compute_reconstruction_error(data, model):
    model.eval() # using trained model
    with torch.no_grad():
        data = data.to(device)
        output = model(data)
        recon_error = torch.mean((data - output) ** 2, dim=[1, 2, 3])
    return recon_error.cpu().detach().numpy()


def plot_reconstructed_images(model, data):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        output = model(data)

    # Visualization of original and reconstructed images (recovered by the model)
    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    for i in range(10):
        axes[0, i].imshow(data[i].cpu().numpy().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title('Original', fontsize=8, pad=2)

        axes[1, i].imshow(output[i].cpu().numpy().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title('Reconstructed', fontsize=8, pad=2)

    plt.suptitle('Original and Reconstructed Malware Images', fontsize=13, y=1.15)
    plt.show()
    

def plot_latent_space(data, model):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        latent = model.encoder(data)  # Assuming the encoder part of the autoencoder is accessible
        latent = latent.view(latent.size(0), -1).cpu().numpy()

    tsne = TSNE(n_components=2)
    latent_2d = tsne.fit_transform(latent)

    plt.figure(figsize=(10, 6))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c='blue', alpha=0.5)
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.title('Latent Space Visualization using TSNE')
    plt.grid(True)
    plt.show()
