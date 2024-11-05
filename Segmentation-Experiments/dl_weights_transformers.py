import os
import torch
import timm
from timm.models import vit_base_patch8_384


def download_pretrained_weights():
    # Créer le dossier si nécessaire
    save_path = "pretrained_models"
    os.makedirs(save_path, exist_ok=True)

    # Charger le modèle pré-entraîné
    model = vit_base_patch8_384(pretrained=True)

    # Sauvegarder les poids
    weights_path = os.path.join(save_path, "vit_base_patch8_384.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Poids sauvegardés dans {weights_path}")


if __name__ == "__main__":
    download_pretrained_weights()
