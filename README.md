# Waterpixels-IMA

## CR réunion du 09/10
Attentes pour le rendu final: savoir expliquer l'implémentation choisie et prendre du recul. Par exemple, le choix du gradient (gradient simple, direction, lab comme dans le papier...) Savoir expliquer les choix sur les paramètres, les algorithmes, avoir fait des tests sur au moins 3-4 images pour tester la robustesse.
- tests à mener:
    - différents gradients
    - influence du paramètre k
    - plusieurs images
    - tester sur les différentes grilles
- améliorations et avancées potentielles: (dans l'ordre de priorité)
    - regarder la notion de transformation en distance
    - faire la transformation en partant des minimats plutôt que des centres des hexagones/carrés
    - coder notre propre watershed
- notes sur la transformation en distance:
    - utile pour les hexagones (actuellement lent)
    - utile pour les composantes connexes (pas encore codé)
    - il sera nécéssaire de tester des distances et trouver "la bonne" (autre test/param sur lequel il faut avoir du recul)
    - création d'une "carte de distances" par passage d'un filtre
- notes sur les réunions:
    - à nous d'organiser
    - prochaine réunion avant le 24 (présentation du rendu?)
    - essayer de faire au moins toutes les deux semaines

Espace lab : https://fr.wikipedia.org/wiki/L*a*b*_CIE_1976


Méthode minima
Binarise ton image selon ce que tu considères comme « objets » vs « fond » (ou selon les contours que tu veux exploiter).
Calcule la transformée de distance chamfer : une passe avant et une passe arrière avec un masque (poids). Masques courants : 3-4 (orthogonaux = 3, diagonaux = 4) ou 5-7 pour une approximation plus précise.
Optionnel : lisser la carte de distance (Gaussian) pour enlever le bruit et rendre les minima plus robustes.
Calculer le gradient de la carte de distance (différences centrales ou Sobel). Calculer la magnitude du gradient.
Trouver minima locaux du champ de gradient (p.ex. pixels où la magnitude est strictement égale au minimum sur un voisinage et en dessous d’un seuil absolu / relatif).
Post-traiter les minima (suppression des très petits composantes, écartement par non-max suppression ou suppression par distance) pour obtenir les points utilisables comme waterpixels.