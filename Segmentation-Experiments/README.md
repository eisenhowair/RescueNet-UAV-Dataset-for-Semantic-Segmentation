# README original

### Train
Single-GPU training is supported. To train the models, modify the settings in `config/rescuenet-pspnet101.yaml` and then run the `train.py`.

### Test
Single-GPU evaluation is supported. To evaluate a model on RescueNet, modify the settings in `config/rescuenet-pspnet101.yaml` and run `test.py`.

## References
This repo is mainly built based on [Official PSPNet](https://github.com/hszhao/semseg), and [Segmenter](https://github.com/rstrudel/segmenter). Thanks for their great work!



# Instructions pour la mise en place du projet

## 1. Ajouter les requirements

- Ajouter les packages suivants à votre environnement :
  - `tensorboardX`
  - `timm==1.0.9` (1.10.0, soit la version actuelle, va poser problème)
  - `einops`

## 2. Télécharger et configurer les modèles

- Télécharger les modèles listés dans `model.py` (au moins `resnet101`, c'est celui utilisé par défaut).
- Renommer le(s) modèle(s) en ajoutant un "_v2" à la fin du nom.
- Placer les modèles renommés dans le dossier `initmodel` (qui doit être créé dans Segmentation-Experiments).

## 3. Préparer le dataset

- Vos dossiers `train/`, `test/`, `val/` doivent contenir 2 dossiers chacuns, un `...-label-img` et un `...-org-img`.
- En vérité, il y a 3 dossiers, mais le dossier `...-label-img` n'est pas utilisé. Il contient les masques colorisés qui sont récupérables via Dropbox, mais le code n'en a pas besoin, seuls  `...-label-img_original` et `...-org-img` sont utilisés. Les masques colorisés peuvent toutefois servir à comparer les résultats obtenus.
- Les datasets nécessaires pouvant être téléchargés (qui iront donc dans les  `...-label-img_original` et `...-org-img`) sont récupérables depuis [ce lien](https://springernature.figshare.com/collections/RescueNet_A_High_Resolution_UAV_Semantic_Segmentation_Benchmark_Dataset_for_Natural_Disaster_Damage_Assessment/6647354/1).


## 4. Optimiser le redimensionnement des images

- Dans le fichier YAML, redimensionner les images à un multiple de 8 + 1 (ex: 17) en cas d'utilisation d'architecture pspnet, ou un nombre divisible par patch_size (variable du fichier YAML) en cas d'utilisation de l'architecture transformer. Un nombre petit permet de gagner du temps lors de l'entraînement (par défaut 713).

## 5. Structure du dépôt

Voilà ce à quoi doit ressembler l'arborescence du projet, sans compter les différents fichiers qui se trouvent dans chacun de ces dossiers :

```bash
└───Segmentation-Experiments
    ├───config
    │   └───michael
    ├───data
    │   └───__pycache__
    ├───dataset
    │   └───RescueNet
    │       ├───test
    │       │   ├───test-label-img
    │       │   ├───test-label-img_original
    │       │   └───test-org-img
    │       ├───train
    │       │   ├───train-label-img
    │       │   ├───train-label-img_original
    │       │   └───train-org-img
    │       └───val
    │           ├───val-label-img
    │           ├───val-label-img_original
    │           └───val-org-img
    ├───exp
    │   └───RescueNet
    │       ├───pspnet101
    │       │   └───model
    │       └───pspnet152
    │           └───model
    ├───initmodel
    ├───metric
    │   └───__pycache__
    ├───models
    │   └───__pycache__
    ├───outputs
    ├───results_saved
    │   ├───color
    │   ├───gray
    │   └───numeric_results
    │       ├───pspnet
    │       └───transformer
    ├───util
    │   └───__pycache__
    └───__pycache__
```

## 6. Divers

- AdamW est implémenté, mais n'a pas été concluant en testant, probablement à cause des paramètres choisis.
- Les fichiers principaux, autour desquels les autres s'articulent, sont `train.py` et `test.py`.
- Tous les tests ont été réalisés sur 10 epochs, peut-être qu'augmenter le nombre fera apparaitre de meilleurs résultats.
- Les architectures PSANet et AttU_Net n'ont été ni testés, ni utilisés. Aussi ne sont-elles pas forcément fonctionnelles.
- Les fichiers contenant les classes ResNet et PSPNet ont été modifiés (`pspnet_for_train.py`,`resnet_for_train.py`), mais leur implémentation initiale, bien que pas utilisée, a été conservée dans les fichiers respectivement nommés `pspnet_original.py` et `resnet_original.py`.
- Pour changer de modèle à utiliser (en tout cas ResNet), il faut modifier le nombre de layers dans le fichier YAML, pour correspondre au modèle ResNet correspondant (101,152, etc.), et dans la partie test du fichier YAML modifier model_path en mettant le nom de l'architecture utilisée pour garder des traces et ne pas enregistrer par dessus un modèle existant qui serait tout à fait différent.
- L'entrainement utilise le poly_learning_rate, défini dans util/util.py (était déjà dans le code préexistant).
- La classe vehicle, avec pool, est celle qui semble poser le plus problème, quel que soit le modèle utilisé.


## 7. Pistes d'exploration

- augmenter decoder_d_model à 512 ou 768
- AdamW plutôt que SGD pour les transformer
- learning rate plus haut uniquement pour le decoder
