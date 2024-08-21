import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from dataset import *
from autoencoder import *
from help_func import *


# Setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(">> My model using", device)

# model 1 : CNN based model
#model = CNNAutoencoder().to(device)
# model 2 : ResNet based model
model = ResNetAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30

# Training
train_losses = []
for epoch in range(num_epochs):
    epoch_loss = 0

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

    train_losses.append(epoch_loss / len(train_loader))

# Plotting the training loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.show()



# Evaluation
# 1. Calculate reconstruction error for normal data
normal_data = next(iter(train_loader))[0]
normal_recon_error = compute_reconstruction_error(normal_data, model)
mean_normal_error = np.mean(normal_recon_error)

# 2. Generate random anomalous data (e.g. noise)
anomalous_data = torch.randn_like(normal_data) * 0.25
anomalous_recon_error = compute_reconstruction_error(anomalous_data, model)
mean_anomalous_error = np.mean(anomalous_recon_error)

# 3. Plotting histograms
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
plt.hist(normal_recon_error, bins=50, alpha=0.95, color='salmon', label='Malware')
plt.hist(anomalous_recon_error, bins=50, alpha=0.95, color='skyblue', label='Non-Malware')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.title('Histogram of Reconstruction Errors')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# 4. Calculate and print mean error rates
mean_normal_error = np.mean(normal_recon_error)
mean_anomalous_error = np.mean(anomalous_recon_error)
print("==========================================================")
print(f">> Mean Normal data Reconstruction Error Rate: {mean_normal_error:.4f}")
print(f">> Mean Anomalous data Reconstruction Error Rate: {mean_anomalous_error:.4f}")
print("==========================================================\n")

# Generate labels (0 for normal, 1 for anomalous)
labels = np.concatenate([np.zeros(len(normal_recon_error)), np.ones(len(anomalous_recon_error))])
scores = np.concatenate([normal_recon_error, anomalous_recon_error])

# 5. Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
print("\n\n")


# Visualization of Reconstructed IMG
plot_reconstructed_images(model, normal_data[:10])
# Example usage
plot_latent_space(next(iter(train_loader))[0], model)
