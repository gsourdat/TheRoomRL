# TheRoomRL

Ce projet implémente un mini-jeu de labyrinthe en Python avec PyGame et un agent d’apprentissage par renforcement (DQN).  
L’agent peut évoluer sur une grille, ramasser des récompenses et éviter des pénalités.

## Structure du projet

- [TheRoom.py](TheRoom.py) : Logique du jeu, gestion de la grille et interface PyGame.  
- [DQNTR.py](DQNTR.py) : Entraînement et définition du réseau de neurones (DQN et variantes).  
- [test.py](test.py) : Script de test pour un modèle sauvegardé.  
- [map.txt](map.txt) : Exemple de configuration de la grille personnalisée.

## Installation

1. Installer Python 3 et pip.  
2. Installer les dépendances :  
   ```bash
   pip install -r requirements.txt
   ```
3. Lancer le jeu :
   ```bash
   python TheRoom.py
   ```
4. Exécuter l’entraînement du DQN :
   ```bash
   python DQNTR.py
   ```
5. Tester un modèle sauvegardé :
   ```bash
   python test.py
   ```

## Utilisation

- Contrôles humains : flèches du clavier pour se déplacer, échappe pour quitter.  
- Entraînement RL : un réseau de neurones estime les Q-values, optimise les poids et enregistre le résultat dans le dossier `models`.

## Licences et remerciements
Ce projet utilise PyGame, PyTorch et d’autres bibliothèques Python open source.