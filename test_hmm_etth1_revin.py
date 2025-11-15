"""
Test HMM con dataset ETTh1 + RevIN.

Pipeline completo RITMO:
1. Cargar ETTh1 con Dataset_ETT_hour (sin StandardScaler)
2. Normalizar con RevIN (módulo normalization/)
3. Entrenar HMM (Baum-Welch)
4. Tokenizar (Viterbi)
5. Validar reconstrucción RevIN
"""

import numpy as np
import matplotlib.pyplot as plt
from data_provider.data_loader import Dataset_ETT_hour
from utils.revin import RevINNormalizer
from hmm import baum_welch, viterbi_decode


class Args:
    """Mock args para Dataset_ETT_hour."""
    augmentation_ratio = 0


def load_etth1(root_path='./dataset/ETT-small',
               data_path='ETTh1.csv',
               target='OT'):
    """
    Carga ETTh1 univariado usando Dataset_ETT_hour.

    Args:
        root_path: Ruta a datasets
        data_path: Archivo CSV
        target: Columna objetivo

    Returns:
        Dict con 'train', 'val', 'test' como arrays [T]
    """
    # Cargar splits SIN StandardScaler
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

    # Extraer datos como arrays [T]
    train = dataset_train.data_x.flatten()
    val = dataset_val.data_x.flatten()
    test = dataset_test.data_x.flatten()

    print(f"   Train: {len(train)} timesteps")
    print(f"   Val:   {len(val)} timesteps")
    print(f"   Test:  {len(test)} timesteps")
    print(f"   Rango train: [{train.min():.2f}, {train.max():.2f}]")
    print(f"   Media train: {train.mean():.2f}, Desv: {train.std():.2f}")

    return {'train': train, 'val': val, 'test': test}


def normalize_data(data_dict):
    """
    Normaliza datos con RevIN.

    Args:
        data_dict: Dict con 'train', 'val', 'test'

    Returns:
        Tupla (data_norm: dict, normalizer: RevINNormalizer)
    """
    normalizer = RevINNormalizer(num_features=1, eps=1e-5, affine=False)

    data_norm = normalizer.fit_transform(
        train_data=data_dict['train'],
        val_data=data_dict['val'],
        test_data=data_dict['test']
    )

    # Mostrar estadísticas
    stats = normalizer.get_statistics('train')
    print(f"\n   RevIN train mean: {stats['mean']:.4f}")
    print(f"   RevIN train stdev: {stats['stdev']:.4f}")
    print(f"   Train norm media: {data_norm['train'].mean():.2e}, desv: {data_norm['train'].std():.4f}")

    return data_norm, normalizer


def validate_revin(data_original, data_normalized, normalizer):
    """
    Valida reconstrucción RevIN.

    Args:
        data_original: Datos originales train
        data_normalized: Datos normalizados train
        normalizer: Instancia de RevINNormalizer

    Returns:
        MSE de reconstrucción
    """
    success, mse = normalizer.validate_reconstruction(
        original=data_original,
        normalized=data_normalized,
        split='train',
        threshold=1e-6
    )

    print("\n" + "="*60)
    print("VALIDACIÓN RECONSTRUCCIÓN RevIN")
    print("="*60)
    print(f"MSE reconstrucción: {mse:.2e}")

    if success:
        print("✓ RevIN norm→denorm funciona PERFECTAMENTE")
    else:
        print(f"✗ Error de reconstrucción: {mse:.2e}")

    return mse


def train_hmm(data_norm, K_values=[3, 5, 7]):
    """
    Entrena HMM con diferentes K.

    Args:
        data_norm: Datos normalizados train
        K_values: Lista de K a probar

    Returns:
        Dict con resultados para cada K
    """
    results = {}

    print("\n" + "="*60)
    print("ENTRENAMIENTO HMM SOBRE ETTh1 (RevIN)")
    print("="*60)

    for K in K_values:
        print(f"\n--- Entrenando con K={K} estados ---")

        # Baum-Welch
        params = baum_welch(
            data_norm,
            K=K,
            max_iter=100,
            epsilon=1e-4,
            random_state=42,
            verbose=True
        )

        # Viterbi
        states_pred, log_likelihood = viterbi_decode(
            data_norm,
            params['A'],
            params['pi'],
            params['mu'],
            params['sigma']
        )

        # Métricas
        state_changes = np.sum(states_pred[1:] != states_pred[:-1]) + 1
        compression_ratio = len(data_norm) / state_changes

        results[K] = {
            'params': params,
            'states': states_pred,
            'log_likelihood': log_likelihood,
            'n_tokens': state_changes,
            'compression_ratio': compression_ratio
        }

        print(f"\nResultados K={K}:")
        print(f"  Log-likelihood: {log_likelihood:.2f}")
        print(f"  Convergió: {params['converged']}")
        print(f"  Iteraciones: {params['n_iter']}")
        print(f"  Tokens generados: {state_changes}")
        print(f"  Ratio compresión: {compression_ratio:.2f}x")
        print(f"  Estados únicos usados: {len(np.unique(states_pred))}/{K}")

    return results


def visualize_results(data_raw, data_norm, results):
    """
    Visualiza resultados HMM.

    Args:
        data_raw: Datos originales train
        data_norm: Datos normalizados train
        results: Dict con resultados de HMM
    """
    n_models = len(results)
    fig, axes = plt.subplots(n_models + 2, 1, figsize=(16, 4*(n_models+2)))

    # Plot 1: Serie original
    axes[0].plot(data_raw, alpha=0.7, linewidth=0.5, color='blue')
    axes[0].set_title('ETTh1 - Oil Temperature (OT) [Univariado]',
                     fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Temperatura (°C)')
    axes[0].grid(alpha=0.3)

    # Plot 2: Serie normalizada
    axes[1].plot(data_norm, alpha=0.7, linewidth=0.5, color='green')
    axes[1].set_title('Serie Normalizada (RevIN)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Valor normalizado')
    axes[1].grid(alpha=0.3)

    # Plots 3+: Estados HMM
    for idx, (K, res) in enumerate(results.items(), start=2):
        states = res['states']
        params = res['params']

        axes[idx].plot(data_norm, alpha=0.3, linewidth=0.5, color='gray', label='Datos')

        # Scatter por estado
        for state_id in range(K):
            mask = states == state_id
            if np.any(mask):
                axes[idx].scatter(
                    np.where(mask)[0],
                    data_norm[mask],
                    c=f'C{state_id}',
                    s=1,
                    alpha=0.6,
                    label=f'Estado {state_id}'
                )

        # Medias de estados
        for state_id in range(K):
            if state_id in states:
                mu = params['mu'][state_id]
                axes[idx].axhline(mu, color=f'C{state_id}', linestyle='--',
                                 alpha=0.5, linewidth=1)

        title = f'Estados HMM (K={K}) | '
        title += f'LL={res["log_likelihood"]:.1f} | '
        title += f'Compresión={res["compression_ratio"]:.1f}x'
        axes[idx].set_title(title, fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Valor norm.')
        axes[idx].grid(alpha=0.3)
        axes[idx].legend(loc='upper right', ncol=K, fontsize=8)

    axes[-1].set_xlabel('Timestep (horas)', fontsize=11)

    plt.tight_layout()
    plt.savefig('hmm_etth1_revin_final.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Gráfico guardado: hmm_etth1_revin_final.png")
    plt.close()


def analyze_compression(results):
    """Analiza métricas de compresión."""
    print("\n" + "="*60)
    print("ANÁLISIS DE COMPRESIÓN - ETTh1")
    print("="*60)

    print("\n{:<10} {:<15} {:<15} {:<15}".format(
        "K", "Log-Likelihood", "Tokens", "Compresión"
    ))
    print("-" * 60)

    for K, res in results.items():
        print("{:<10} {:<15.2f} {:<15} {:<15.2f}x".format(
            K,
            res['log_likelihood'],
            res['n_tokens'],
            res['compression_ratio']
        ))

    best_K = max(results.keys(), key=lambda k: results[k]['log_likelihood'])
    print(f"\n✓ Mejor modelo (log-likelihood): K={best_K}")
    print(f"  Log-likelihood: {results[best_K]['log_likelihood']:.2f}")
    print(f"  Compresión: {results[best_K]['compression_ratio']:.2f}x")


def analyze_parameters(results, K_focus=3):
    """Analiza parámetros aprendidos del HMM."""
    if K_focus not in results:
        K_focus = list(results.keys())[0]

    print("\n" + "="*60)
    print(f"PARÁMETROS APRENDIDOS HMM (K={K_focus})")
    print("="*60)

    params = results[K_focus]['params']
    order = np.argsort(params['mu'])

    print("\nEstados (ordenados por μ):")
    print("{:<10} {:<15} {:<15} {:<15}".format(
        "Estado", "μ (media)", "σ (desv)", "Persistencia"
    ))
    print("-" * 60)

    for state_id in order:
        mu = params['mu'][state_id]
        sigma = params['sigma'][state_id]
        persistence = params['A'][state_id, state_id]

        print("{:<10} {:<15.3f} {:<15.3f} {:<15.2%}".format(
            state_id, mu, sigma, persistence
        ))

    print("\nMatriz de Transición A:")
    print(params['A'])

    print("\nDistribución Inicial π:")
    print(params['pi'])


if __name__ == "__main__":
    print("="*60)
    print("TEST HMM - ETTh1 + RevIN (MÓDULO LIMPIO)")
    print("="*60)

    # 1. Cargar datos
    print("\n1. Cargando ETTh1 con Dataset_ETT_hour (scale=False)...")
    data = load_etth1()

    # 2. Normalizar con RevIN
    print("   Normalizando con RevIN (utils/revin.py)...")
    data_norm, normalizer = normalize_data(data)

    # 3. Validar reconstrucción
    validate_revin(data['train'], data_norm['train'], normalizer)

    # 4. Entrenar HMM
    K_values = [3, 5, 7]
    results = train_hmm(data_norm['train'], K_values=K_values)

    # 5. Visualizar
    print("\n2. Generando visualizaciones...")
    visualize_results(data['train'], data_norm['train'], results)

    # 6. Análisis
    analyze_compression(results)
    analyze_parameters(results, K_focus=3)

    print("\n" + "="*60)
    print("✓ TEST COMPLETADO")
    print("="*60)
    print("\nResultados guardados en: hmm_etth1_revin_final.png")
    print("\nPipeline RITMO validado:")
    print("  1. ✓ Dataset_ETT_hour (scale=False)")
    print("  2. ✓ RevIN normalización (utils/revin.py)")
    print("  3. ✓ HMM entrenado (hmm/baum_welch.py)")
    print("  4. ✓ Viterbi tokenización (hmm/viterbi.py)")
    print("  5. ✓ RevIN denorm validado")
    print("\nPróximos pasos:")
    print("  - Fase 2: Embeddings estructurados e_k = [μ_k, σ_k, A[k,:]]")
    print("  - Fase 3: Integración con Transformer para predicción")
    print("="*60 + "\n")
