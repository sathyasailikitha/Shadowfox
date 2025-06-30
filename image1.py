import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from PIL import Image
import random

# 1. Configuration 
data_dir = '/home/likitha/Downloads/shadowfox/dataset'  
external_image_path = '/home/likitha/Downloads/shadowfox/image.jpg'  
resize_size = (32, 32)
batch_size = 64
num_epochs = 5

# 2. Image transforms 
transform = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 3. Load dataset 
full_dataset = ImageFolder(root=data_dir, transform=transform)
classes = full_dataset.classes
num_classes = len(classes)

# Split into train/test
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_data, test_data = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 4. Define CNN model
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 8 * 8)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 5. Setup device, model, optimizer 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CustomCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Train the model
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.3f}")

#  7. Evaluate the model 
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nAccuracy on your dataset: {accuracy:.2f}%")

# 8. Show one sample test image 
sample_idx = random.randint(0, len(test_data) - 1)
img, true_label = test_data[sample_idx]
img_display = img * 0.5 + 0.5  # Unnormalize
img = img.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img)
    _, predicted = torch.max(output, 1)

predicted_class = classes[predicted.item()]
true_class = classes[true_label]

plt.imshow(img_display.permute(1, 2, 0))
plt.title(f"Predicted: {predicted_class}, True: {true_class}")
plt.axis('off')
plt.show()

#  9. External image testing
print("\n🔍 Testing your own external image...")
try:
    img_ext = Image.open(external_image_path).convert('RGB')
    transform_external = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_tensor_ext = transform_external(img_ext).unsqueeze(0).to(device)

    with torch.no_grad():
        output_ext = model(img_tensor_ext)
        _, predicted_ext = torch.max(output_ext, 1)
        predicted_ext_class = classes[predicted_ext.item()]

    print(f"\n Your external image is predicted as: **{predicted_ext_class.upper()}**")
    plt.imshow(img_ext)
    plt.title(f"Predicted: {predicted_ext_class}")
    plt.axis('off')
    plt.show()

except Exception as e:
    print(" Error loading external image:", e)

