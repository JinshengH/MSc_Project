# Your graph
def plot_loss_acc(metrics, num_epochs):
    epochs = range(1, num_epochs + 1)
    
    fig = plt.figure(figsize=(12,5))
    
    # Loss
    plt.subplot(1,2,1)
    plt.plot(epochs, metrics["train_loss"], label='Train Loss')
    plt.plot(epochs, metrics["val_loss"], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    
    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs, metrics["train_acc"], label='Train Accuracy')
    plt.plot(epochs, metrics["val_acc"], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    fig.savefig(ResultPath + "training_loss.svg")
    plt.show()

