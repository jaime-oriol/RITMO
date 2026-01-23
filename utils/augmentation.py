"""
Data augmentation para series temporales.
Implementa técnicas de aumento de datos basadas en:
- Jitter, Scaling, Rotation, Permutation (Um et al., 2017)
- Magnitude/Time Warping (Le Guennec et al., 2016)
- Window Slicing/Warping (Le Guennec et al., 2016)
- SPAWNER, WDBA, Guided Warping (métodos basados en DTW)
"""

import numpy as np
from tqdm import tqdm


def jitter(x, sigma=0.03):
    """
    Añade ruido gaussiano a la señal.
    Simula variabilidad en mediciones. sigma controla intensidad.
    """
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=0.1):
    """
    Escala aleatoriamente cada feature por factor gaussiano.
    Simula cambios de ganancia en sensores.
    """
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0], x.shape[2]))
    return np.multiply(x, factor[:, np.newaxis, :])


def rotation(x):
    """
    Invierte signo aleatorio y permuta canales.
    Útil para datos IMU donde orientación no importa.
    """
    x = np.array(x)
    flip = np.random.choice([-1, 1], size=(x.shape[0], x.shape[2]))  # Flip aleatorio
    rotate_axis = np.arange(x.shape[2])
    np.random.shuffle(rotate_axis)  # Permutar canales
    return flip[:, np.newaxis, :] * x[:, :, rotate_axis]


def permutation(x, max_segments=5, seg_mode="equal"):
    """
    Permuta segmentos de la serie temporal.
    Mantiene patrones locales pero cambia orden global.

    max_segments: Número máximo de segmentos
    seg_mode: 'equal' (segmentos iguales) o 'random' (puntos aleatorios)
    """
    orig_steps = np.arange(x.shape[1])
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                # Puntos de corte aleatorios
                split_points = np.random.choice(x.shape[1] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                # Segmentos de igual tamaño
                splits = np.array_split(orig_steps, num_segs[i])
            # Permutar orden de segmentos
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret


def magnitude_warp(x, sigma=0.2, knot=4):
    """
    Deforma magnitud usando curva spline suave.
    Multiplica serie por spline aleatorio para variar amplitud localmente.

    knot: Número de puntos de control del spline
    sigma: Desviación de factores en puntos de control
    """
    from scipy.interpolate import CubicSpline

    orig_steps = np.arange(x.shape[1])
    # Factores aleatorios en puntos de control
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    # Posiciones de puntos de control (uniformemente espaciados)
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        # Interpolar spline para cada dimensión
        warper = np.array([CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps)
                          for dim in range(x.shape[2])]).T
        ret[i] = pat * warper  # Multiplicar serie por factor variable

    return ret


def time_warp(x, sigma=0.2, knot=4):
    """
    Deforma eje temporal usando curva spline suave.
    Acelera/desacelera partes de la serie de forma no lineal.

    knot: Número de puntos de control
    sigma: Intensidad de la deformación
    """
    from scipy.interpolate import CubicSpline

    orig_steps = np.arange(x.shape[1])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            # Calcular nuevo mapeado temporal
            time_warp = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim])(orig_steps)
            scale = (x.shape[1] - 1) / time_warp[-1]  # Normalizar a longitud original
            # Interpolar valores en nuevo eje temporal
            ret[i, :, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[1] - 1), pat[:, dim]).T
    return ret


def window_slice(x, reduce_ratio=0.9):
    """
    Extrae ventana aleatoria y la estira a tamaño original.
    Simula cambios de velocidad/frecuencia.

    reduce_ratio: Proporción de la serie a extraer (0.9 = 90%)
    """
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio * x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x

    # Puntos de inicio aleatorios para cada muestra
    starts = np.random.randint(low=0, high=x.shape[1] - target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            # Interpolar ventana extraída a longitud original
            ret[i, :, dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]),
                                       np.arange(target_len),
                                       pat[starts[i]:ends[i], dim]).T
    return ret


def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    """
    Escala temporalmente una ventana aleatoria.
    Comprime (scale<1) o expande (scale>1) porción de la serie.

    window_ratio: Proporción de serie a deformar
    scales: Factores de escala posibles
    """
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])  # Escala por muestra
    warp_size = np.ceil(window_ratio * x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)

    # Posiciones aleatorias de ventana
    window_starts = np.random.randint(low=1, high=x.shape[1] - warp_size - 1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i], dim]  # Antes de ventana
            # Ventana escalada
            window_seg = np.interp(np.linspace(0, warp_size - 1, num=int(warp_size * warp_scales[i])),
                                   window_steps, pat[window_starts[i]:window_ends[i], dim])
            end_seg = pat[window_ends[i]:, dim]  # Después de ventana
            # Concatenar y reinterpollar a longitud original
            warped = np.concatenate((start_seg, window_seg, end_seg))
            ret[i, :, dim] = np.interp(np.arange(x.shape[1]),
                                       np.linspace(0, x.shape[1] - 1., num=warped.size), warped).T
    return ret


def spawner(x, labels, sigma=0.05, verbose=0):
    """
    SPAWNER: Promedia patrones de misma clase usando DTW.
    Crea muestras sintéticas interpolando entre ejemplos similares.

    labels: Etiquetas de clase para cada muestra
    sigma: Jitter añadido al resultado
    verbose: -1=sin warnings, 1=mostrar gráficos
    """
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6983028/
    import utils.dtw as dtw

    random_points = np.random.randint(low=1, high=x.shape[1] - 1, size=x.shape[0])
    window = np.ceil(x.shape[1] / 10.).astype(int)  # Ventana DTW
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels  # One-hot a índices

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        # Seleccionar otra muestra de misma clase
        choices = np.delete(np.arange(x.shape[0]), i)
        choices = np.where(l[choices] == l[i])[0]

        if choices.size > 0:
            random_sample = x[np.random.choice(choices)]
            # DTW en dos segmentos (antes/después de punto aleatorio)
            path1 = dtw.dtw(pat[:random_points[i]], random_sample[:random_points[i]],
                           dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
            path2 = dtw.dtw(pat[random_points[i]:], random_sample[random_points[i]:],
                           dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
            combined = np.concatenate((np.vstack(path1), np.vstack(path2 + random_points[i])), axis=1)
            # Promediar patrones alineados
            mean = np.mean([pat[combined[0]], random_sample[combined[1]]], axis=0)
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(orig_steps,
                                          np.linspace(0, x.shape[1] - 1., num=mean.shape[0]), mean[:, dim]).T
        else:
            ret[i, :] = pat

    return jitter(ret, sigma=sigma)  # Añadir ruido


def wdba(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True, verbose=0):
    """
    WDBA: Weighted DTW Barycentric Averaging.
    Calcula baricentro ponderado de patrones de misma clase.

    batch_size: Prototipos a promediar
    slope_constraint: Restricción DTW ('symmetric' o 'asymmetric')
    use_window: Si usar ventana Sakoe-Chiba
    """
    # https://ieeexplore.ieee.org/document/8215569
    x = np.array(x)
    import utils.dtw as dtw

    window = np.ceil(x.shape[1] / 10.).astype(int) if use_window else None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

    ret = np.zeros_like(x)
    for i in range(ret.shape[0]):
        choices = np.where(l == l[i])[0]
        if choices.size > 0:
            k = min(choices.size, batch_size)
            random_prototypes = x[np.random.choice(choices, k, replace=False)]

            # Calcular matriz de distancias DTW
            dtw_matrix = np.zeros((k, k))
            for p, prototype in enumerate(random_prototypes):
                for s, sample in enumerate(random_prototypes):
                    if p != s:
                        dtw_matrix[p, s] = dtw.dtw(prototype, sample, dtw.RETURN_VALUE,
                                                   slope_constraint=slope_constraint, window=window)

            # Encontrar medoide (mínima distancia total)
            medoid_id = np.argsort(np.sum(dtw_matrix, axis=1))[0]
            nearest_order = np.argsort(dtw_matrix[medoid_id])
            medoid_pattern = random_prototypes[medoid_id]

            # Promedio ponderado (más peso a patrones cercanos)
            average_pattern = np.zeros_like(medoid_pattern)
            weighted_sums = np.zeros((medoid_pattern.shape[0]))

            for nid in nearest_order:
                if nid == medoid_id or dtw_matrix[medoid_id, nearest_order[1]] == 0.:
                    average_pattern += medoid_pattern
                    weighted_sums += np.ones_like(weighted_sums)
                else:
                    path = dtw.dtw(medoid_pattern, random_prototypes[nid], dtw.RETURN_PATH,
                                   slope_constraint=slope_constraint, window=window)
                    dtw_value = dtw_matrix[medoid_id, nid]
                    warped = random_prototypes[nid, path[1]]
                    # Peso exponencial: más cercano = más peso
                    weight = np.exp(np.log(0.5) * dtw_value / dtw_matrix[medoid_id, nearest_order[1]])
                    average_pattern[path[0]] += weight * warped
                    weighted_sums[path[0]] += weight

            ret[i, :] = average_pattern / weighted_sums[:, np.newaxis]
        else:
            ret[i, :] = x[i]

    return ret


def random_guided_warp(x, labels, slope_constraint="symmetric", use_window=True, dtw_type="normal", verbose=0):
    """
    Random Guided Warp: Alinea cada patrón a otro aleatorio de su clase.
    Deforma eje temporal según alineamiento DTW.

    dtw_type: 'normal' (DTW estándar) o 'shape' (shapeDTW)
    """
    import utils.dtw as dtw

    window = np.ceil(x.shape[1] / 10.).astype(int) if use_window else None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        choices = np.delete(np.arange(x.shape[0]), i)
        choices = np.where(l[choices] == l[i])[0]

        if choices.size > 0:
            random_prototype = x[np.random.choice(choices)]

            # Calcular path de alineamiento
            if dtw_type == "shape":
                path = dtw.shape_dtw(random_prototype, pat, dtw.RETURN_PATH,
                                     slope_constraint=slope_constraint, window=window)
            else:
                path = dtw.dtw(random_prototype, pat, dtw.RETURN_PATH,
                              slope_constraint=slope_constraint, window=window)

            # Aplicar warping según path
            warped = pat[path[1]]
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(orig_steps,
                                          np.linspace(0, x.shape[1] - 1., num=warped.shape[0]), warped[:, dim]).T
        else:
            ret[i, :] = pat

    return ret


def random_guided_warp_shape(x, labels, slope_constraint="symmetric", use_window=True):
    """Wrapper: Random Guided Warp usando shapeDTW."""
    return random_guided_warp(x, labels, slope_constraint, use_window, dtw_type="shape")


def discriminative_guided_warp(x, labels, batch_size=6, slope_constraint="symmetric",
                               use_window=True, dtw_type="normal", use_variable_slice=True, verbose=0):
    """
    Discriminative Guided Warp: Alinea a prototipo que maximiza separación inter-clase.
    Selecciona prototipo que esté más cerca de su clase y más lejos de otras.

    use_variable_slice: Si aplicar window slice proporcional al warp
    """
    import utils.dtw as dtw

    window = np.ceil(x.shape[1] / 10.).astype(int) if use_window else None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

    positive_batch = np.ceil(batch_size / 2).astype(int)
    negative_batch = np.floor(batch_size / 2).astype(int)

    ret = np.zeros_like(x)
    warp_amount = np.zeros(x.shape[0])

    for i, pat in enumerate(x):
        choices = np.delete(np.arange(x.shape[0]), i)
        positive = np.where(l[choices] == l[i])[0]   # Misma clase
        negative = np.where(l[choices] != l[i])[0]   # Otras clases

        if positive.size > 0 and negative.size > 0:
            pos_k = min(positive.size, positive_batch)
            neg_k = min(negative.size, negative_batch)
            positive_prototypes = x[np.random.choice(positive, pos_k, replace=False)]
            negative_prototypes = x[np.random.choice(negative, neg_k, replace=False)]

            # Calcular scores discriminativos
            pos_aves = np.zeros((pos_k))
            neg_aves = np.zeros((pos_k))

            if dtw_type == "shape":
                dtw_func = dtw.shape_dtw
            else:
                dtw_func = dtw.dtw

            for p, pos_prot in enumerate(positive_prototypes):
                # Distancia promedio a positivos
                for ps, pos_samp in enumerate(positive_prototypes):
                    if p != ps:
                        pos_aves[p] += (1. / (pos_k - 1.)) * dtw_func(pos_prot, pos_samp, dtw.RETURN_VALUE,
                                                                      slope_constraint=slope_constraint, window=window)
                # Distancia promedio a negativos
                for ns, neg_samp in enumerate(negative_prototypes):
                    neg_aves[p] += (1. / neg_k) * dtw_func(pos_prot, neg_samp, dtw.RETURN_VALUE,
                                                          slope_constraint=slope_constraint, window=window)

            # Seleccionar prototipo más discriminativo
            selected_id = np.argmax(neg_aves - pos_aves)  # Lejos de negativo, cerca de positivo
            path = dtw_func(positive_prototypes[selected_id], pat, dtw.RETURN_PATH,
                           slope_constraint=slope_constraint, window=window)

            # Aplicar warping
            warped = pat[path[1]]
            warp_path_interp = np.interp(orig_steps, np.linspace(0, x.shape[1] - 1., num=warped.shape[0]), path[1])
            warp_amount[i] = np.sum(np.abs(orig_steps - warp_path_interp))

            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(orig_steps,
                                          np.linspace(0, x.shape[1] - 1., num=warped.shape[0]), warped[:, dim]).T
        else:
            ret[i, :] = pat
            warp_amount[i] = 0.

    # Variable slicing proporcional al warp
    if use_variable_slice:
        max_warp = np.max(warp_amount)
        if max_warp == 0:
            ret = window_slice(ret, reduce_ratio=0.9)
        else:
            for i, pat in enumerate(ret):
                ret[i] = window_slice(pat[np.newaxis, :, :],
                                      reduce_ratio=0.9 + 0.1 * warp_amount[i] / max_warp)[0]

    return ret


def discriminative_guided_warp_shape(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True):
    """Wrapper: Discriminative Guided Warp usando shapeDTW."""
    return discriminative_guided_warp(x, labels, batch_size, slope_constraint, use_window, dtw_type="shape")


def run_augmentation(x, y, args):
    """
    Ejecuta pipeline de augmentation múltiple.
    Aplica augmentation_ratio veces y concatena resultados.
    """
    print("Augmenting %s" % args.data)
    np.random.seed(args.seed)

    x_aug = x
    y_aug = y

    if args.augmentation_ratio > 0:
        augmentation_tags = "%d" % args.augmentation_ratio
        for n in range(args.augmentation_ratio):
            x_temp, augmentation_tags = augment(x, y, args)
            x_aug = np.append(x_aug, x_temp, axis=0)  # Añadir datos aumentados
            y_aug = np.append(y_aug, y, axis=0)       # Duplicar etiquetas
            print("Round %d: %s done" % (n, augmentation_tags))
        if args.extra_tag:
            augmentation_tags += "_" + args.extra_tag
    else:
        augmentation_tags = args.extra_tag

    return x_aug, y_aug, augmentation_tags


def run_augmentation_single(x, y, args):
    """
    Ejecuta augmentation sobre serie individual o batch.
    Adapta dimensiones automáticamente.
    """
    np.random.seed(args.seed)

    x_aug = x
    y_aug = y

    if len(x.shape) < 3:
        # Serie individual: [T, C] → [1, T, C]
        x_input = x[np.newaxis, :]
    elif len(x.shape) == 3:
        # Batch: mantener [B, T, C]
        x_input = x
    else:
        raise ValueError("Input must be (batch_size, sequence_length, num_channels) dimensional")

    if args.augmentation_ratio > 0:
        augmentation_tags = "%d" % args.augmentation_ratio
        for n in range(args.augmentation_ratio):
            x_aug, augmentation_tags = augment(x_input, y, args)
        if args.extra_tag:
            augmentation_tags += "_" + args.extra_tag
    else:
        augmentation_tags = args.extra_tag

    if len(x.shape) < 3:
        x_aug = x_aug.squeeze(0)  # Revertir a [T, C]

    return x_aug, y_aug, augmentation_tags


def augment(x, y, args):
    """
    Aplica augmentaciones según flags en args.
    Cada flag activa una técnica específica.
    """
    import utils.augmentation as aug

    augmentation_tags = ""

    # Augmentaciones básicas
    if args.jitter:
        x = aug.jitter(x)
        augmentation_tags += "_jitter"
    if args.scaling:
        x = aug.scaling(x)
        augmentation_tags += "_scaling"
    if args.rotation:
        x = aug.rotation(x)
        augmentation_tags += "_rotation"

    # Permutaciones
    if args.permutation:
        x = aug.permutation(x)
        augmentation_tags += "_permutation"
    if args.randompermutation:
        x = aug.permutation(x, seg_mode="random")
        augmentation_tags += "_randomperm"

    # Warpings
    if args.magwarp:
        x = aug.magnitude_warp(x)
        augmentation_tags += "_magwarp"
    if args.timewarp:
        x = aug.time_warp(x)
        augmentation_tags += "_timewarp"

    # Window-based
    if args.windowslice:
        x = aug.window_slice(x)
        augmentation_tags += "_windowslice"
    if args.windowwarp:
        x = aug.window_warp(x)
        augmentation_tags += "_windowwarp"

    # DTW-based
    if args.spawner:
        x = aug.spawner(x, y)
        augmentation_tags += "_spawner"
    if args.dtwwarp:
        x = aug.random_guided_warp(x, y)
        augmentation_tags += "_rgw"
    if args.shapedtwwarp:
        x = aug.random_guided_warp_shape(x, y)
        augmentation_tags += "_rgws"
    if args.wdba:
        x = aug.wdba(x, y)
        augmentation_tags += "_wdba"
    if args.discdtw:
        x = aug.discriminative_guided_warp(x, y)
        augmentation_tags += "_dgw"
    if args.discsdtw:
        x = aug.discriminative_guided_warp_shape(x, y)
        augmentation_tags += "_dgws"

    return x, augmentation_tags
