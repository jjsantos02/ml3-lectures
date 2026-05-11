"""Utility functions defined for RNN Fundamentals II"""

import matplotlib.pyplot as plt


def plot_history(history, metric='mae'):
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
    accuracy = history.history[metric]
    loss = history.history['loss']
    
    # Generate epoch number list
    epochs = range(1, len(accuracy) + 1)

    # Initialize figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Plot epochs and accuracy/loss values
    axes[0].plot(epochs, accuracy, 'o', color='tab:blue',
                 label='Train MAE')
    axes[1].plot(epochs, loss, 'o', color='tab:orange', label='Train Loss')

    # Add axis labels and legends
    for ax in axes:
        ax.set_xlabel("Epochs")
        ax.legend()
    axes[0].set_ylabel("MAE")
    axes[1].set_ylabel("Loss")
    fig.suptitle("Fitting History Plots", fontsize=16, weight='bold')

    return fig, axes
