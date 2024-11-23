from unet import UNet

from dataloader import PokemonDataset

from torch.utils.data import DataLoader

import torch

torch.cuda.empty_cache()

train_data  = PokemonDataset('~/akshat/dip_project/dip/train/Masks','~/akshat/dip_project/dip/train/Images')
validation_data  = PokemonDataset('~/akshat/dip_project/dip/validation/Masks','~/akshat/dip_project/dip/validation/Images')
test_data  = PokemonDataset('~/akshat/dip_project/dip/test/Masks','~/akshat/dip_project/dip/test/Images')

train_dataloader = DataLoader(train_data, batch_size=9, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
validation_dataloader = DataLoader(validation_data,batch_size=1, shuffle=True)

model = UNet(dim=1,n_filters=100,FL=3,init= "he_normal",drop=0.10,lmbda=0.00001).to("cuda")

loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

n_samples = train_data.__len__()
   
runningLoss = []

# nb_epoch = 5

device = 'cuda'


from tqdm import tqdm  # For progress bars during training

def normalize(image):
    n_img = (image-torch.min(image))/(torch.max(image)-torch.min(image))
    return n_img

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    
#    print(type(dataloader))
    for images, masks in tqdm(dataloader):
        images = normalize(images)
        masks = normalize(masks)
        masks = masks.unsqueeze(1)
        images, masks = images.to(torch.float32).to(device), masks.to(torch.float32).to(device)

        optimizer.zero_grad()  # Zero gradients
        
#        images.to(torch.float32)
        # Forward pass
#        print(type(images)
#        ,"\n",
#        images)
        
        outputs = model(images)
#        print(outputs.shape)
#        print(masks.shape)
        loss = criterion()(outputs, masks)  # Calculate the loss

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Track running loss and accuracy (simplified)
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += masks.numel()  # Total number of pixels
        correct += (predicted == masks).sum().item()  # Correct predictions

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total * 100
    return avg_loss, accuracy

# Function to validate the model
def validate_epoch(model, dataloader, criterion):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  
        for images, masks in tqdm(dataloader):
            images = normalize(images)
            masks = normalize(masks)
            masks = masks.unsqueeze(0)
            images, masks = images.to(device), masks.to(device)

            
            outputs = model(images)
            loss = criterion()(outputs, masks)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += masks.numel()  
            correct += (predicted == masks).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total * 100
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        
        val_loss, val_accuracy = validate_epoch(model, val_loader, criterion)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")


        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), '~/akshat/best_model_MSE_20_epochs.pt')
        print("-" * 50)

    print("Training complete.")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")

def test_model(model, test_loader):
    model.eval()  
    correct = 0
    total = 0
    total_loss=0
    with torch.no_grad():
        for images, masks in tqdm(test_loader):
            images = normalize(images)
            masks = normalize(masks)
            masks= masks.unsqueeze(0)
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_loss = criterion()(outputs,masks)
            total_loss = total_loss+test_loss.item()
            total += masks.numel()
            correct += (predicted == masks).sum().item()

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    average_loss = total_loss/len(test_loader)
    print(f"Test Loss: {average_loss:.2f}%")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs=20

#criterion=torch.nn.BCELoss

criterion = torch.nn.MSELoss

train_model(model, train_dataloader, validation_dataloader, criterion, optimizer, num_epochs)

model.load_state_dict(torch.load('~/akshat/best_model_MSE_20_epochs.pt'))
test_model(model, test_dataloader)

