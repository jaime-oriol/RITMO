"""
Dynamic Time Warping (DTW) para alineamiento de series temporales.
Implementa DTW clásico y versión acelerada usando scipy.cdist.
Usado para métricas de evaluación y data augmentation.
"""

from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist
from math import isinf


def dtw(x, y, dist, warp=1, w=inf, s=1.0):
    """
    Calcula Dynamic Time Warping entre dos secuencias.

    DTW encuentra alineamiento óptimo que minimiza distancia acumulada,
    permitiendo deformaciones temporales (comprimir/estirar).

    Args:
        x: Secuencia 1, shape [N1, M] (N1 timesteps, M features)
        y: Secuencia 2, shape [N2, M]
        dist: Función de distancia entre elementos
        warp: Número máximo de pasos diagonales por movimiento
        w: Ancho de ventana Sakoe-Chiba (inf = sin restricción)
        s: Peso para movimientos no-diagonales (s>1 penaliza warping)

    Returns:
        Tupla (distancia_minima, matriz_costo, costo_acumulado, path)
        - path: (array_i, array_j) índices del camino óptimo
    """
    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))  # Ventana debe permitir alinear
    assert s > 0

    r, c = len(x), len(y)  # Longitudes de secuencias

    # Inicializar matriz de costos acumulados
    if not isinf(w):
        # Con ventana: infinito fuera de banda
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        # Sin ventana: solo primera fila/columna infinito
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf

    D1 = D0[1:, 1:]  # Vista sin padding

    # Calcular matriz de costos locales
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[i], y[j])

    C = D1.copy()  # Guardar costos locales

    # Programación dinámica: acumular costos mínimos
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            # Opciones: diagonal, horizontal (warp y), vertical (warp x)
            min_list = [D0[i, j]]  # Diagonal (sin penalización)
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]  # Penalizar warping
            D1[i, j] += min(min_list)

    # Reconstruir camino óptimo
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)

    return D1[-1, -1], C, D1, path


def accelerated_dtw(x, y, dist, warp=1):
    """
    DTW acelerado usando scipy.cdist para matriz de distancias.

    Más eficiente que dtw() básico porque scipy.cdist usa
    implementaciones optimizadas (C/Fortran) para métricas comunes.

    Args:
        x: Secuencia 1, shape [N1, M]
        y: Secuencia 2, shape [N2, M]
        dist: String de métrica scipy o función callable
            Strings válidos: 'euclidean', 'cityblock', 'cosine', etc.
        warp: Número máximo de pasos diagonales

    Returns:
        Tupla (distancia_minima, matriz_costo, costo_acumulado, path)
    """
    assert len(x)
    assert len(y)

    # Asegurar 2D para cdist
    if ndim(x) == 1:
        x = x.reshape(-1, 1)
    if ndim(y) == 1:
        y = y.reshape(-1, 1)

    r, c = len(x), len(y)

    # Inicializar matriz de costos acumulados
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf

    D1 = D0[1:, 1:]

    # Calcular todas las distancias de una vez con cdist
    D0[1:, 1:] = cdist(x, y, dist)
    C = D1.copy()

    # Programación dinámica
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                min_list += [D0[min(i + k, r), j],
                             D0[i, min(j + k, c)]]
            D1[i, j] += min(min_list)

    # Reconstruir camino
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)

    return D1[-1, -1], C, D1, path


def _traceback(D):
    """
    Reconstruye camino óptimo mediante backtracking.

    Empieza en esquina inferior-derecha y retrocede
    hacia (0,0) eligiendo mínimo en cada paso.

    Args:
        D: Matriz de costos acumulados [r+1, c+1]

    Returns:
        Tupla (array_i, array_j) con índices del camino
    """
    i, j = array(D.shape) - 2  # Empezar en última celda válida
    p, q = [i], [j]  # Listas para almacenar camino

    while (i > 0) or (j > 0):
        # Elegir dirección de mínimo costo
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            # Diagonal: ambos avanzan
            i -= 1
            j -= 1
        elif tb == 1:
            # Vertical: solo x avanza
            i -= 1
        else:
            # Horizontal: solo y avanza
            j -= 1
        p.insert(0, i)
        q.insert(0, j)

    return array(p), array(q)


# Codigo de prueba
if __name__ == '__main__':
    w = inf
    s = 1.0

    if 1:  # Test 1-D numérico
        from sklearn.metrics.pairwise import manhattan_distances
        x = [0, 0, 1, 1, 2, 4, 2, 1, 2, 0]
        y = [1, 1, 1, 2, 2, 2, 2, 3, 2, 0]
        dist_fun = manhattan_distances
        w = 1

    elif 0:  # Test 2-D numérico
        from sklearn.metrics.pairwise import euclidean_distances
        x = [[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [4, 3], [2, 3], [1, 1], [2, 2], [0, 1]]
        y = [[1, 0], [1, 1], [1, 1], [2, 1], [4, 3], [4, 3], [2, 3], [3, 1], [1, 2], [1, 0]]
        dist_fun = euclidean_distances

    else:  # Test con strings
        from nltk.metrics.distance import edit_distance
        x = ['i', 'soon', 'found', 'myself', 'muttering', 'to', 'the', 'walls']
        y = ['see', 'drown', 'himself']
        dist_fun = edit_distance

    dist, cost, acc, path = dtw(x, y, dist_fun, w=w, s=s)

    # Visualizar resultado
    from matplotlib import pyplot as plt
    plt.imshow(cost.T, origin='lower', cmap=plt.cm.Reds, interpolation='nearest')
    plt.plot(path[0], path[1], '-o')  # Camino óptimo
    plt.xticks(range(len(x)), x)
    plt.yticks(range(len(y)), y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('tight')

    if isinf(w):
        plt.title('Minimum distance: {}, slope weight: {}'.format(dist, s))
    else:
        plt.title('Minimum distance: {}, window widht: {}, slope weight: {}'.format(dist, w, s))
    plt.show()
