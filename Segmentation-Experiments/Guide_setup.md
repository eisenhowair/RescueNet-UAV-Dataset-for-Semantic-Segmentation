# Instructions pour la mise en place du projet

## 1. Ajouter les requirements

- Ajouter les packages suivants à votre environnement :
  - `tensorboardX`
  - `timm==1.0.9` (1.10.0, soit la version actuelle, va poser problème)
  - `einops`

## 2. Configurer les fichiers de configuration

- Créer un dossier `michael/` dans `config/`.
- Déplacer le fichier `rescuenet-pspnet101.yaml` dans le dossier `config/michael/`.

## 3. Télécharger et configurer les modèles

- Télécharger les modèles listés dans `model.py` (au moins `resnet101`, c'est celui utilisé par défaut).
- Renommer le(s) modèle(s) en ajoutant un "_v2" à la fin du nom.
- Placer les modèles renommés dans le dossier `initmodel` (qui doit être crée dans Segmentation-Experiments).

## 4. Modifier les imports dans le code

- Dans les fichiers `train.py` et `test.py`, remplacer l'import suivant :
  ```python
  from data import RescueNetV2 as dataset
  ```
  par :
  ```python
  from data.rescuenet import RescueNet as dataset
  ```

## 5. Préparer le dataset

- Créer un dossier `dataset` dans Segmentation-Experiments.
- À l'intérieur de `dataset`, créer un sous-dossier `RescueNet`.
- Dans `RescueNet`, créer deux sous-dossiers : `train/` et `test/` (`val/` aussi, mais peut-être pas obligatoire).
- Placer les deux dossiers contenus dans le dataset respectif dans `train/` et `test/` (idem pour `val`).
- Vos dossiers `train/`, `test/`, `val/` doivent contenir 2 dossier chacuns, un `...-label-img` et un `...-org-img`.
- Les datasets peuvent être téléchargés depuis [ce lien](https://springernature.figshare.com/collections/RescueNet_A_High_Resolution_UAV_Semantic_Segmentation_Benchmark_Dataset_for_Natural_Disaster_Damage_Assessment/6647354/1).

## 6. Ajouter la barre de progression (facultatif)

- Ajouter `tqdm` dans `train.py` pour suivre les epochs.

## 7. Sauvegarder les modèles

- Créer les dossiers du chemin de la variable `save_path` dans le fichier YAML.

## 8. Optimiser le redimensionnement des images

- Dans le fichier YAML, redimensionner les images à un multiple de 8 + 1 (ex: 17) pour gagner du temps lors de l'entraînement (par défaut 713).

## 9. Initialiser les variables d'entraînement

- Au début de la fonction `main_worker()` dans `train.py`, ajouter les lignes suivantes pour initialiser les variables :
  ```python
  train_epochs = []
  train_loss = []
  train_accuracy = []

  val_epochs = []
  val_loss = []
  val_accuracy = []
  ```
## 10. Corriger test.py

- Mettre en commentaire dans `test.py`:
  - `colors = np.loadtxt(color_folder).astype('uint8')`
  - `names = [line.rstrip('\n') for line in open(args.names_path)]`
- Dans la variable `model_path` du fichier YAML, mettre le chemin vers un modèle (normalement enregistré dans
  `exp/RescueNet/pspnet101/model/train_epoch_10.pth` par exemple)
- La fonction test.py peut être modifiée pour pouvoir récupérer les résultats numériques

## 11 Utiliser des poids préentrainés

- Modifier le fichier `resnet.py` pour mettre deep_base à False afin d'avoir les dimensions conformes du domaine
- Modifier le fichier `pspnet.py` pour mettre à jour la classe PSPNet.
- Mettre `user_pretrained_weights` à True devrait alors fonctionner.
  
Suivez ces étapes pour correctement mettre en place et préparer votre projet pour l'entraînement.
Si un problème par rapport à une variable survient, notamment `args`, d'abord regarder le fichier YAML.


images de taille 401x401, entrainé (a duré 9 heures) sur 10 epochs
Entrainement essayé sur des images de 713*713, mais une epoch prend plus de 20 heures
9 minutes par epoch sur des images 241*241
9 images par epoch sur des images 281*281
Sur des images de 561, une epoch prend 20 heures