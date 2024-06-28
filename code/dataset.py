from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Get dataset and preprocessing
# 1.Define transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure images are grayscale
    transforms.Resize((28, 28)),  # Resize images to 28 x 28
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 2.Load MalwarePix Dataset
path = './data/img/malwarePix_small'
train_dataset = datasets.ImageFolder(root=path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)

# 3.Check the data
for images, labels in train_loader:
    print(f'Batch size: {images.shape[0]}')
    print(f'Image shape: {images.shape[1:]}')
    print(f'Labels: {labels[:10]}')
    break

# Print MalwarePix Dataset
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.show()

# Get a batch of training data
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Show images
imshow(utils.make_grid(images))

# Print labels
print(' '.join('%5s' % labels[j].item() for j in range(8)))
