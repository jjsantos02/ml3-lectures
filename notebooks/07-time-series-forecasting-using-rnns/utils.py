"""Utility functions defined for TSF using RNNs notebook"""

import matplotlib.pyplot as plt


def plot_history(history):
    """Generate the fitting history plot given the history object from
    model training

    Generates the training and validation accuracy and loss plots.

    Parameters
    ----------
    history : keras History object
        History object returned during model fitting

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
        Figure and axes of the generated validation and accuracy loss
        plots
    """
    # Get accuracies and losses from history object
    accuracy = history.history['mae']
    loss = history.history['loss']
    val_accuracy = history.history['val_mae']
    val_loss = history.history['val_loss']
    
    # Generate epoch number list
    epochs = range(1, len(accuracy) + 1)

    # Initialize figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Plot epochs and accuracy/loss values
    axes[0].plot(epochs, accuracy, 'o', color='tab:blue',
                 label='Train MAE')
    axes[0].plot(epochs, val_accuracy, '--', color='tab:blue',
                 label='Validation MAE')
    axes[1].plot(epochs, loss, 'o', color='tab:orange', label='Train Loss')
    axes[1].plot(epochs, val_loss, '--', color='tab:orange',
                 label='Validation Loss')

    # Add axis labels and legends
    for ax in axes:
        ax.set_xlabel("Epochs")
        ax.legend()
    axes[0].set_ylabel("MAE")
    axes[1].set_ylabel("Loss")
    fig.suptitle("Training History Plots", fontsize=16, weight='bold')

    return fig, axes
