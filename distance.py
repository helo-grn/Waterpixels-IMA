import numpy as np

def d(p, q):
    return np.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)

def naive_distance(im, Q, sigma):
    w, h = im.shape[:2]
    dist_im = np.zeros((w, h))
    for row in range(0, h):
        for col in range(0, w):
            p = (col, row)
            min_dist = np.inf
            for q in Q:
                if d(p, q) - min_dist < 0:
                    min_dist = d(p, q)
            dist_im[col, row] = 2/sigma * min_dist
    return dist_im

def chamfer_distance_transform(image_binaire, d_h=3, d_d=4):
    """
    Calcule la carte de distance de Chanfrein (3-4) à partir d'une image binaire.

    Les marqueurs (pixels d'intérêt) doivent être initialisés à 0,
    et le reste de l'image à 'infini' (une grande valeur).

    Args:
        image_binaire (np.ndarray): Image binaire (marqueurs à 0, fond à 'infini').
        d_h (int): Coût de déplacement horizontal/vertical (voisinage 4-connexe). Par défaut à 3.
        d_d (int): Coût de déplacement diagonal (voisinage 8-connexe). Par défaut à 4.

    Returns:
        np.ndarray: La carte de distance normalisée.
    """
    H, W = image_binaire.shape
    carte_distance = np.full((H, W), np.inf)
    
    # Initialisation : les marqueurs (0) ont distance 0
    carte_distance[image_binaire == 0] = 0

    # Normalisation pour la distance 3-4: la division finale par 3 est implicite
    # dans le choix des poids, mais on peut la normaliser à la fin.

    # --- 1er Balayage : Gauche -> Droite / Haut -> Bas (propagation 'directe') ---
    # Masque avant (voisinage supérieur et gauche) :
    # 4  3  4
    # 3  .
    # .  .  .

    for i in range(1, H):
        for j in range(1, W):
            # Voisins à considérer: Haut-Gauche, Haut, Haut-Droite (si < W-1), Gauche
            
            # Gestion du bord droit pour Haut-Droite
            cost_hd = d_d if j < W - 1 else float('inf')

            min_prev = min(
                carte_distance[i - 1, j - 1] + d_d,  # Haut-Gauche (d_d)
                carte_distance[i - 1, j] + d_h,      # Haut (d_h)
                carte_distance[i, j - 1] + d_h,      # Gauche (d_h)
                carte_distance[i, j]                 # Valeur actuelle
            )
            
            # On inclut le Haut-Droite uniquement si on n'est pas sur le bord droit
            if j < W - 1:
                min_prev = min(min_prev, carte_distance[i - 1, j + 1] + d_d)
            
            carte_distance[i, j] = min_prev

        # Balayage sur la première ligne (i=0) pour les voisins Gauche uniquement
        # (déjà initialisé si marqueur, ou 'inf')
        if i == 1:
            for j in range(1, W):
                carte_distance[0, j] = min(carte_distance[0, j], carte_distance[0, j - 1] + d_h)


    # --- 2ème Balayage : Droite -> Gauche / Bas -> Haut (propagation 'rétrograde') ---
    # Masque après (voisinage inférieur et droit) :
    # .  .  .
    # .  .  3
    # 4  3  4

    for i in range(H - 2, -1, -1):
        for j in range(W - 2, -1, -1):
            # Voisins à considérer: Bas-Droite, Bas, Bas-Gauche (si > 0), Droite
            
            # Gestion du bord gauche pour Bas-Gauche
            cost_bg = d_d if j > 0 else float('inf')
            
            min_prev = min(
                carte_distance[i + 1, j + 1] + d_d,  # Bas-Droite (d_d)
                carte_distance[i + 1, j] + d_h,      # Bas (d_h)
                carte_distance[i, j + 1] + d_h,      # Droite (d_h)
                carte_distance[i, j]                 # Valeur actuelle
            )
            
            # On inclut le Bas-Gauche uniquement si on n'est pas sur le bord gauche
            if j > 0:
                min_prev = min(min_prev, carte_distance[i + 1, j - 1] + d_d)

            carte_distance[i, j] = min_prev

        # Balayage sur la dernière ligne (i=H-1) pour les voisins Droite uniquement
        # (déjà fait, on ne refait que les colonnes)
        if i == H - 2:
            for j in range(W - 2, -1, -1):
                carte_distance[H - 1, j] = min(carte_distance[H - 1, j], carte_distance[H - 1, j + 1] + d_h)


    # Normalisation finale par max(d_h, d_d) pour se rapprocher de l'unité (ici par 3)
    return carte_distance / d_h # Normalisation par la distance horizontale/verticale