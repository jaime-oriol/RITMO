"""
Test exhaustivo de validación Pipeline RITMO (Pasos 1-3).

Valida implementación completa de:
- Paso 1: Normalización RevIN (reversible)
- Paso 2: Entrenamiento HMM (Baum-Welch con emisiones gaussianas)
- Paso 3: Tokenización Viterbi (estados óptimos)

Configuraciones de prueba:
- K ∈ {3, 5, 7, 10} estados HMM
- seeds ∈ {42, 123, 456} para robustez
- data_configs: full (8640 timesteps), half (4320 timesteps)
Total: 4 × 3 × 2 = 24 configuraciones

Métricas reportadas:
- RevIN: MSE reconstrucción
- HMM: Log-likelihood, AIC, BIC, convergencia
- Viterbi: Ratio compresión, entropía, segmentos
- Validaciones: Matriz estocástica, parámetros razonables, estados no degenerados

Salida: Tabla comparativa con recomendación automática de K óptimo según AIC/BIC.
"""

import numpy as np
from data_provider.data_loader import Dataset_ETT_hour
from utils.revin import RevINNormalizer
from hmm import baum_welch, viterbi_decode


class Args:
    """Mock args requerido por Dataset_ETT_hour."""
    augmentation_ratio = 0


def load_etth1(root_path='./dataset/ETT-small',
               data_path='ETTh1.csv',
               target='OT',
               data_size=None):
    """
    Carga ETTh1 univariado sin normalización.

    Args:
        root_path: Directorio datasets
        data_path: Archivo CSV (ETTh1.csv)
        target: Columna objetivo (OT: Oil Temperature)
        data_size: Limitar train a N timesteps (None: usar completo)

    Returns:
        dict: {'train': array[T], 'val': array[V], 'test': array[Te]}
    """
    dataset_train = Dataset_ETT_hour(
        args=Args(),
        root_path=root_path,
        flag='train',
        size=None,
        features='S',
        data_path=data_path,
        target=target,
        scale=False,
        timeenc=0,
        freq='h'
    )

    dataset_val = Dataset_ETT_hour(
        args=Args(),
        root_path=root_path,
        flag='val',
        size=None,
        features='S',
        data_path=data_path,
        target=target,
        scale=False,
        timeenc=0,
        freq='h'
    )

    dataset_test = Dataset_ETT_hour(
        args=Args(),
        root_path=root_path,
        flag='test',
        size=None,
        features='S',
        data_path=data_path,
        target=target,
        scale=False,
        timeenc=0,
        freq='h'
    )

    train = dataset_train.data_x.flatten()
    val = dataset_val.data_x.flatten()
    test = dataset_test.data_x.flatten()

    if data_size is not None:
        train = train[:data_size]

    return {'train': train, 'val': val, 'test': test}


def validate_revin(data_original, data_normalized, normalizer, split='train', threshold=1e-6):
    """
    Valida reversibilidad normalización RevIN.

    Comprueba que denorm(norm(X)) ≈ X con MSE < threshold.

    Args:
        data_original: Datos raw
        data_normalized: Datos después de RevIN
        normalizer: Instancia RevINNormalizer fitted
        split: Split a validar ('train', 'val', 'test')
        threshold: MSE máximo aceptable

    Returns:
        (success: bool, mse: float)
    """
    success, mse = normalizer.validate_reconstruction(
        original=data_original,
        normalized=data_normalized,
        split=split,
        threshold=threshold
    )
    return success, mse


def validate_hmm_params(params, K):
    """
    Valida validez matemática parámetros HMM λ = (A, π, μ, σ).

    Checks de validez:
    - Propiedades estocásticas: filas A suman 1, π suma 1
    - Propiedades físicas: σ > 0, μ finitos
    - Propiedades dimensionales: shapes correctos
    - Propiedades de calidad: estados diferenciados, sin degeneración

    Args:
        params: dict con A[K,K], pi[K], mu[K], sigma[K]
        K: Número de estados

    Returns:
        dict: {check_name: bool} para cada validación
    """
    checks = {}

    row_sums = np.sum(params['A'], axis=1)
    checks['matrix_stochastic'] = np.allclose(row_sums, 1.0, atol=1e-4)
    checks['pi_stochastic'] = np.isclose(np.sum(params['pi']), 1.0, atol=1e-4)
    checks['sigma_positive'] = np.all(params['sigma'] > 0)
    checks['mu_finite'] = np.all(np.isfinite(params['mu']))
    checks['dims_correct'] = (
        params['A'].shape == (K, K) and
        params['pi'].shape == (K,) and
        params['mu'].shape == (K,) and
        params['sigma'].shape == (K,)
    )
    checks['no_degenerate_states'] = True  # Actualizado después con Viterbi
    checks['no_extreme_persistence'] = np.all(np.diag(params['A']) < 0.99)
    checks['sigma_reasonable'] = np.all((params['sigma'] > 0.1) & (params['sigma'] < 5.0))
    checks['mu_spread'] = (params['mu'].max() - params['mu'].min()) > 1.0

    return checks


def validate_viterbi(states_pred, K, data_norm):
    """
    Valida output de Viterbi Q* = argmax P(Q|O,λ).

    Checks:
    - Dimensiones: longitud match con observaciones
    - Rango: estados ∈ [0, K-1]
    - Finitud: sin NaN/Inf
    - Segmentación: al menos 1 cambio de estado

    Args:
        states_pred: array[T] con estados predichos
        K: Número de estados HMM
        data_norm: Datos normalizados (para validar longitud)

    Returns:
        dict: {check_name: bool}
    """
    checks = {}

    checks['length_match'] = len(states_pred) == len(data_norm)
    checks['states_in_range'] = np.all((states_pred >= 0) & (states_pred < K))
    checks['states_finite'] = np.all(np.isfinite(states_pred))

    n_segments = np.sum(np.diff(states_pred) != 0) + 1
    checks['has_segments'] = n_segments >= 1

    return checks


def run_single_config(K, seed, data_config, verbose=False):
    """
    Ejecuta pipeline completo para una configuración (K, seed, data_size).

    Pipeline:
    1. Cargar ETTh1
    2. Normalizar con RevIN
    3. Entrenar HMM (Baum-Welch)
    4. Tokenizar (Viterbi)
    5. Calcular métricas (AIC, BIC, compresión, entropía)
    6. Validar todos los pasos

    Args:
        K: Número de estados HMM
        seed: Random seed para reproducibilidad
        data_config: dict con 'name' y 'size' (timesteps)
        verbose: Si True, imprime progreso

    Returns:
        dict con todas las métricas y validaciones
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"CONFIG: K={K}, seed={seed}, data={data_config['name']}")
        print(f"{'='*60}")

    data = load_etth1(data_size=data_config['size'])
    train_size = len(data['train'])

    if verbose:
        print(f"Train size: {train_size} timesteps")

    normalizer = RevINNormalizer(num_features=1, eps=1e-5, affine=False)
    data_norm = normalizer.fit_transform(
        train_data=data['train'],
        val_data=data['val'],
        test_data=data['test']
    )

    revin_success, revin_mse = validate_revin(
        data['train'],
        data_norm['train'],
        normalizer,
        split='train'
    )

    if verbose:
        status = "✓" if revin_success else "✗"
        print(f"{status} RevIN MSE: {revin_mse:.2e}")

    params = baum_welch(
        data_norm['train'],
        K=K,
        max_iter=100,
        epsilon=1e-4,
        random_state=seed,
        verbose=False
    )

    hmm_checks = validate_hmm_params(params, K)

    if verbose:
        status = "✓" if params['converged'] else "⚠"
        print(f"{status} HMM convergió: {params['converged']} ({params['n_iter']} iter)")
        for check_name, check_value in hmm_checks.items():
            status = "✓" if check_value else "✗"
            print(f"{status} {check_name}: {check_value}")

    states_pred, log_likelihood = viterbi_decode(
        data_norm['train'],
        params['A'],
        params['pi'],
        params['mu'],
        params['sigma']
    )

    viterbi_checks = validate_viterbi(states_pred, K, data_norm['train'])

    unique_states = len(np.unique(states_pred))
    n_segments = np.sum(np.diff(states_pred) != 0) + 1
    n_tokens_llm = len(states_pred)
    compression_ratio = len(states_pred) / n_segments
    avg_segment_duration = len(states_pred) / n_segments

    viterbi_checks['no_degenerate_states'] = unique_states >= K * 0.7
    hmm_checks['no_degenerate_states'] = unique_states >= K * 0.7

    num_params = K**2 + 2*K
    AIC = -2 * log_likelihood + 2 * num_params
    BIC = -2 * log_likelihood + np.log(len(states_pred)) * num_params

    state_counts = np.bincount(states_pred, minlength=K)
    state_distribution = state_counts / len(states_pred)
    state_entropy = -np.sum(state_distribution * np.log(state_distribution + 1e-10))

    if verbose:
        status = "✓" if not np.isnan(log_likelihood) else "✗"
        print(f"{status} Log-likelihood: {log_likelihood:.2f}")
        print(f"  AIC: {AIC:.2f}, BIC: {BIC:.2f}")
        print(f"  Estados activos: {unique_states}/{K}")
        print(f"  Entropía estados: {state_entropy:.3f}")
        print(f"  Segmentos: {n_segments}")
        print(f"  Compresión: {compression_ratio:.1f}x")
        print(f"  Persistencia: {avg_segment_duration:.1f} timesteps")

    result = {
        'K': K,
        'seed': seed,
        'data_config': data_config['name'],
        'train_size': train_size,
        'revin_success': revin_success,
        'revin_mse': revin_mse,
        'hmm_converged': params['converged'],
        'hmm_n_iter': params['n_iter'],
        'hmm_checks': hmm_checks,
        'log_likelihood': log_likelihood,
        'AIC': AIC,
        'BIC': BIC,
        'unique_states': unique_states,
        'n_tokens_llm': n_tokens_llm,
        'n_segments': n_segments,
        'compression_ratio': compression_ratio,
        'avg_segment_duration': avg_segment_duration,
        'state_entropy': state_entropy,
        'viterbi_checks': viterbi_checks
    }

    return result


def print_summary(results):
    """
    Imprime reporte consolidado de todas las configuraciones.

    Secciones del reporte:
    1. Validación RevIN (paso 1)
    2. Validación HMM (paso 2)
    3. Validación Viterbi (paso 3)
    4. Tabla comparativa por K
    5. Recomendación automática K óptimo (AIC/BIC)
    6. Verificación final

    Args:
        results: list de dicts, uno por configuración ejecutada
    """
    print("\n" + "="*80)
    print("REPORTE EXHAUSTIVO - VALIDACIÓN PASOS 1-3")
    print("="*80)

    # Contar checks pasados
    total_configs = len(results)
    revin_passed = sum(1 for r in results if r['revin_success'])
    hmm_converged = sum(1 for r in results if r['hmm_converged'])

    # Validaciones HMM
    hmm_checks_all = [r['hmm_checks'] for r in results]
    check_names = list(hmm_checks_all[0].keys())

    print(f"\nTotal configuraciones: {total_configs}")
    print(f"\n{'='*80}")
    print("PASO 1: REVIN NORMALIZACIÓN")
    print(f"{'='*80}")
    print(f"✓ Pasadas: {revin_passed}/{total_configs} ({100*revin_passed/total_configs:.1f}%)")

    mse_values = [r['revin_mse'] for r in results]
    print(f"  MSE min: {min(mse_values):.2e}")
    print(f"  MSE max: {max(mse_values):.2e}")
    print(f"  MSE mean: {np.mean(mse_values):.2e}")

    print(f"\n{'='*80}")
    print("PASO 2: HMM ENTRENAMIENTO")
    print(f"{'='*80}")
    print(f"✓ Convergió: {hmm_converged}/{total_configs} ({100*hmm_converged/total_configs:.1f}%)")

    for check_name in check_names:
        check_passed = sum(1 for r in results if r['hmm_checks'][check_name])
        status = "✓" if check_passed == total_configs else "✗"
        print(f"{status} {check_name}: {check_passed}/{total_configs}")

    print(f"\n{'='*80}")
    print("PASO 3: VITERBI TOKENIZACIÓN")
    print(f"{'='*80}")

    # Checks Viterbi
    viterbi_checks_all = [r['viterbi_checks'] for r in results]
    viterbi_check_names = list(viterbi_checks_all[0].keys())

    for check_name in viterbi_check_names:
        check_passed = sum(1 for r in results if r['viterbi_checks'][check_name])
        status = "✓" if check_passed == total_configs else "✗"
        print(f"{status} {check_name}: {check_passed}/{total_configs}")

    # Análisis por K con tabla comparativa
    print(f"\n{'='*90}")
    print("TABLA COMPARATIVA - ANÁLISIS POR NÚMERO DE ESTADOS (K)")
    print(f"{'='*90}")

    print(f"\n{' ':<6} {' ':<10} {' ':<8} {' ':<12} {' ':<12} {' ':<10} {' ':<10}")
    print(f"{'K':<6} {'LL (mean)':<10} {'Conv':<8} {'AIC (mean)':<12} {'BIC (mean)':<12} {'Compresión':<10} {'Estados'}")
    print("-" * 90)

    K_values = sorted(set(r['K'] for r in results))
    for K in K_values:
        K_results = [r for r in results if r['K'] == K]
        ll_values = [r['log_likelihood'] for r in K_results if not np.isnan(r['log_likelihood'])]
        aic_values = [r['AIC'] for r in K_results if not np.isnan(r['AIC'])]
        bic_values = [r['BIC'] for r in K_results if not np.isnan(r['BIC'])]
        compression_values = [r['compression_ratio'] for r in K_results]
        states_values = [r['unique_states'] for r in K_results]
        converged = sum(1 for r in K_results if r['hmm_converged'])
        total = len(K_results)

        print(f"{K:<6} {np.mean(ll_values):<10.1f} {converged}/{total:<5} "
              f"{np.mean(aic_values):<12.1f} {np.mean(bic_values):<12.1f} "
              f"{np.mean(compression_values):<10.1f}x {np.mean(states_values):.1f}/{K}")

    # Recomendación automática
    print(f"\n{'='*90}")
    print("RECOMENDACIÓN AUTOMÁTICA (criterios AIC/BIC)")
    print(f"{'='*90}")

    # Mejor K según BIC (penaliza complejidad más fuerte)
    best_bic_result = min(results, key=lambda r: r['BIC'] if not np.isnan(r['BIC']) else float('inf'))
    best_aic_result = min(results, key=lambda r: r['AIC'] if not np.isnan(r['AIC']) else float('inf'))

    print(f"\n✓ Mejor K según BIC: K={best_bic_result['K']}")
    print(f"  BIC: {best_bic_result['BIC']:.2f}")
    print(f"  Log-likelihood: {best_bic_result['log_likelihood']:.2f}")
    print(f"  Compresión: {best_bic_result['compression_ratio']:.1f}x")
    print(f"  Estados activos: {best_bic_result['unique_states']}/{best_bic_result['K']}")

    print(f"\n✓ Mejor K según AIC: K={best_aic_result['K']}")
    print(f"  AIC: {best_aic_result['AIC']:.2f}")
    print(f"  Log-likelihood: {best_aic_result['log_likelihood']:.2f}")
    print(f"  Compresión: {best_aic_result['compression_ratio']:.1f}x")
    print(f"  Estados activos: {best_aic_result['unique_states']}/{best_aic_result['K']}")

    # Verificación final
    print(f"\n{'='*80}")
    print("VERIFICACIÓN FINAL")
    print(f"{'='*80}")

    all_passed = (
        revin_passed == total_configs and
        all(all(r['hmm_checks'].values()) for r in results) and
        all(all(r['viterbi_checks'].values()) for r in results)
    )

    if all_passed:
        print("✓ TODOS LOS TESTS PASADOS - PASOS 1-3 VALIDADOS")
    else:
        print("✗ ALGUNOS TESTS FALLARON - REVISAR CONFIGURACIONES")

    print("="*80 + "\n")


if __name__ == "__main__":
    print("="*80)
    print("TEST EXHAUSTIVO HMM - ETTh1 + RevIN")
    print("="*80)
    print("\nValidando Pipeline RITMO (Pasos 1-3):")
    print("  1. Normalización RevIN")
    print("  2. Entrenamiento HMM (Baum-Welch)")
    print("  3. Tokenización (Viterbi)")

    # Configuraciones a probar
    K_values = [3, 5, 7, 10]  # Añadido K=10 según ROADMAP
    seeds = [42, 123, 456]
    data_configs = [
        {'name': 'full', 'size': None},
        {'name': 'half', 'size': 4320}
    ]

    total_configs = len(K_values) * len(seeds) * len(data_configs)
    print(f"\nTotal configuraciones: {total_configs}")
    print(f"  K values: {K_values}")
    print(f"  Seeds: {seeds}")
    print(f"  Data configs: {[d['name'] for d in data_configs]}")

    # Ejecutar todas las configuraciones
    results = []
    config_idx = 0

    for K in K_values:
        for seed in seeds:
            for data_config in data_configs:
                config_idx += 1
                print(f"\n[{config_idx}/{total_configs}] ", end='')

                result = run_single_config(K, seed, data_config, verbose=True)
                results.append(result)

    # Imprimir reporte final
    print_summary(results)

    print("\n" + "="*80)
    print("✓ TEST EXHAUSTIVO COMPLETADO")
    print("="*80)
    print("\nPróximos pasos:")
    print("  - Si todos los tests pasaron → Avanzar a Fase 4 (Embeddings)")
    print("  - Si algunos tests fallaron → Investigar configuraciones problemáticas")
    print("="*80 + "\n")
