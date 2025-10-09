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