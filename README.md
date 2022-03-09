1. il est conseillé de créer un nouvel environnement python pour éviter les conflits sinon effectuer uniquement les étape 3. 5. et 6. :

`python3 -m venv ./PyP7/`

2. activer l'environnement :

`source ./PyP7/bin/activate`

3. installer les dépendances:

`pip install -r requiremements.txt`

4. ajouter le noyau pour jupyter :

`./PyP7/bin/python -m ipykernel install --user --name 'PyP7'`

5. ouvrir le notebook

`jupyter notebook ./P7_01_notebook.ipynb`

6. changer le noyau dans jupyter : / Noyau / Changer le noyau / PyP7
