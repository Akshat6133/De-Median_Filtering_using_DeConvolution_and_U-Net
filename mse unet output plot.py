 
import matplotlib.pyplot as plt

# Epoch data
epochs = list(range(1, 21))

# Training metrics
train_loss = [0.0092, 0.0036, 0.0028, 0.0024, 0.0021, 0.0019, 0.0018, 0.0017, 0.0016, 0.0015,
              0.0015, 0.0014, 0.0014, 0.0013, 0.0013, 0.0012, 0.0012, 0.0012, 0.0011, 0.0011]
# train_accuracy = [0.89] * 20

# Validation metrics
val_loss = [0.0025, 0.0015, 0.0012, 0.0011, 0.0011, 0.0011, 0.0010, 0.0010, 0.0009, 0.0009,
            0.0009, 0.0009, 0.0008, 0.0008, 0.0008, 0.0008, 0.0007, 0.0007, 0.0007, 0.0007]
# val_accuracy = [0.04] * 20

# Plotting loss and accuracy for training and validation
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plotting loss
ax1.plot(epochs, train_loss, label="Train Loss", color='blue', marker='o')
ax1.plot(epochs, val_loss, label="Validation Loss", color='orange', marker='o')
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.set_title("Model Training and Validation Metrics")
ax1.legend(loc='upper left')

# Creating a second y-axis for accuracy
ax2 = ax1.twinx()
# ax2.plot(epochs, train_accuracy, label="Train Accuracy", color='green', linestyle='--')
# ax2.plot(epochs, val_accuracy, label="Validation Accuracy", color='red', linestyle='--')
# ax2.set_ylabel("Accuracy (%)")

# Combining legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='center right')

plt.grid()
plt.show()
