import matplotlib.pyplot as plt
import pandas as pd

# Read CSV file containing training logs
df = pd.read_csv("/maps/projects/renlab/people/nkz325/00_reformer_test_run/Reformer/M0_722_e10/training_log.csv")

# Extract training metrics data
epochs = df['epoch']
# epochs = range(1, 7)  # Alternative: manual epoch range if needed
train_loss = df['train_loss']
val_loss = df['val_loss']
val_auc = df['val_auc']
train_auc = df['train_auc']

# Plot AUC curves (Training vs Validation)
plt.figure(figsize=(8, 6))
# plt.plot(epochs, train_loss, '-o', label='Train Loss', color='blue', linewidth=2)
# plt.plot(epochs, val_loss, '-o', label='Val Loss', color='red', linewidth=2)
plt.plot(epochs, train_auc, '-o', label='Train AUC', color='blue', linewidth=2)
plt.plot(epochs, val_auc, '-o', label='Val AUC', color='red', linewidth=2)

# Add vertical lines for each epoch to show training checkpoints
for epoch in epochs:
    plt.axvline(x=epoch, color='purple', linestyle='--', linewidth=0.5)

# Configure plot appearance
plt.title("Training vs Validation AUC", fontsize=14)
# plt.title("Training vs Validation Loss", fontsize=14)  # Alternative title for loss plot
# plt.xlabel("Epoch", fontsize=12)
# plt.ylabel("Loss", fontsize=12)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("AUC", fontsize=12)
plt.xticks(epochs)  # Set x-axis ticks to match epoch values
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("auc_curve.png", dpi=300)  # Save high-resolution AUC plot
plt.show()


# Plot Loss curves (Training vs Validation)
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_loss, '-o', label='Train Loss', color='blue', linewidth=2)
plt.plot(epochs, val_loss, '-o', label='Val Loss', color='red', linewidth=2)
# plt.plot(epochs, train_auc, '-o', label='Train AUC', color='blue', linewidth=2)
# plt.plot(epochs, val_auc, '-o', label='Val AUC', color='red', linewidth=2)

# Add vertical lines for each epoch to show training checkpoints
for epoch in epochs:
    plt.axvline(x=epoch, color='purple', linestyle='--', linewidth=0.5)

# Configure plot appearance
plt.title("Training vs Validation Loss", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.xticks(epochs)  # Set x-axis ticks to match epoch values
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Loss_curve.png", dpi=300)  # Save high-resolution loss plot
plt.show()