import matplotlib.pyplot as plt
import torch

def show_tensor_image(tensor, title=None):
    """
    Visualiza un tensor de forma (3, H, W) como imagen RGB.
    El tensor debe estar en rango [0, 1] o normalizado previamente.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Se esperaba un tensor de PyTorch.")
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("El tensor debe tener forma (3, H, W).")

    # Convertir a formato (H, W, C) para matplotlib
    img = tensor.permute(1, 2, 0).cpu().numpy()

    # Clipping por si hay valores fuera de [0, 1]
    img = img.clip(0, 1)

    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
