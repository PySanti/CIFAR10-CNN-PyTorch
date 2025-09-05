import matplotlib.pyplot as plt

def plot_performance(train_losses, val_losses):
    """
    Plotea la pérdida de entrenamiento y validación por época.

    Args:
        train_losses (list or array): Lista de pérdidas de entrenamiento.
        val_losses (list or array): Lista de pérdidas de validación.
        save_path (str): Ruta opcional para guardar la imagen.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'r-', label='Pérdida entrenamiento')
    plt.plot(epochs, val_losses, 'b-', label='Pérdida validación')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)
    plt.show()
