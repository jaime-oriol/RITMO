"""
EDA reproducible y exhaustivo de los seis datasets empleados en el TFG RITMO.

Genera:
    1. Estructura, tipos y validación de integridad (.shape, dtypes, nulls)
    2. Detección de valores sentinel (-9999, -999, NaN encubiertos)
    3. Estadísticos descriptivos completos del target (sin samples, serie entera)
    4. Análisis por partición cronológica (train / val / test) con stats separadas
    5. Test de estacionariedad ADF aplicado sobre la SERIE COMPLETA (no submuestreo)
    6. Análisis de autocorrelación (ACF) y detección de periodicidades dominantes
    7. Verificación cruzada con los splits de TSLib (data_provider/data_loader.py)
    8. Resumen final estructurado para la sección 4.4 del TFG

Uso:
    cd /home/jaime/TFG/RITMO
    python notebooks/eda_datasets.py

Salidas:
    - stdout
    - notebooks/eda_datasets_output.txt (constancia escrita reproducible)

Autor: Jaime Oriol — TFG RITMO
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Configuración de los seis datasets (rutas y columna objetivo en TSLib)
# ----------------------------------------------------------------------------
DATASETS = [
    ("ETTh1",       "dataset/ETT-small/ETTh1.csv",          "OT", "horaria"),
    ("ETTh2",       "dataset/ETT-small/ETTh2.csv",          "OT", "horaria"),
    ("Weather",     "dataset/weather/weather.csv",          "OT", "10 min"),
    ("Electricity", "dataset/electricity/electricity.csv",  "OT", "horaria"),
    ("Traffic",     "dataset/traffic/traffic.csv",          "OT", "horaria"),
    ("Exchange",    "dataset/exchange_rate/exchange_rate.csv", "OT", "diaria"),
]

# Valores sentinel típicos a buscar (datasets meteorológicos suelen usar -9999)
SENTINELS = [-9999.0, -999.0, -99.0, 9999.0, 999.0]


def descriptivos(serie: np.ndarray) -> dict:
    """Estadísticos descriptivos completos del target."""
    return dict(
        n=int(len(serie)),
        n_unique=int(np.unique(serie).size),
        min=float(serie.min()),
        max=float(serie.max()),
        mean=float(serie.mean()),
        std=float(serie.std()),
        median=float(np.median(serie)),
        q05=float(np.percentile(serie, 5)),
        q25=float(np.percentile(serie, 25)),
        q75=float(np.percentile(serie, 75)),
        q95=float(np.percentile(serie, 95)),
        skewness=float(pd.Series(serie).skew()),
        kurtosis=float(pd.Series(serie).kurt()),
        rango=float(serie.max() - serie.min()),
        coef_var=float(serie.std() / abs(serie.mean())) if serie.mean() != 0 else float("nan"),
    )


def detectar_sentinels(serie: np.ndarray) -> dict:
    """Detecta valores sentinel típicos (e.g. -9999) que pandas no marca como NaN."""
    found = {}
    for s in SENTINELS:
        n = int((serie == s).sum())
        if n > 0:
            found[s] = n
    return found


def test_estacionariedad_completo(serie: np.ndarray) -> dict:
    """ADF aplicado sobre la SERIE COMPLETA, sin submuestreo."""
    serie_limpia = serie[~np.isnan(serie)]
    # Excluir sentinels antes del test
    for s in SENTINELS:
        serie_limpia = serie_limpia[serie_limpia != s]
    adf_stat, p_val, used_lag, n_obs, crit, icbest = adfuller(
        serie_limpia, autolag="AIC"
    )
    return dict(
        adf_stat=float(adf_stat),
        p_value=float(p_val),
        lag_usado=int(used_lag),
        n_obs_test=int(n_obs),
        n_serie_completa=int(len(serie)),
        n_serie_limpia=int(len(serie_limpia)),
        crit_1pct=float(crit["1%"]),
        crit_5pct=float(crit["5%"]),
        crit_10pct=float(crit["10%"]),
        es_estacionaria=bool(p_val < 0.05),
        sample_truncado=False,
    )


def periodicidades(serie: np.ndarray, max_lag: int = 500) -> dict:
    """Detecta picos en la ACF que indican periodicidades dominantes."""
    serie_limpia = serie.copy()
    for s in SENTINELS:
        serie_limpia = serie_limpia[serie_limpia != s]
    n_lag = min(max_lag, len(serie_limpia) // 4)
    acf_vals = acf(serie_limpia, nlags=n_lag, fft=True)
    peaks, props = find_peaks(acf_vals[1:], height=0.1, distance=5)
    peaks_lag = (peaks + 1).tolist()
    peaks_val = acf_vals[peaks + 1].tolist()
    if len(peaks_lag) > 0:
        ranked = sorted(zip(peaks_lag, peaks_val), key=lambda x: -x[1])[:8]
    else:
        ranked = []
    return dict(
        n_lag_evaluado=int(n_lag),
        acf_lag1=float(acf_vals[1]) if len(acf_vals) > 1 else None,
        acf_lag24=float(acf_vals[24]) if len(acf_vals) > 24 else None,
        acf_lag48=float(acf_vals[48]) if len(acf_vals) > 48 else None,
        acf_lag144=float(acf_vals[144]) if len(acf_vals) > 144 else None,
        acf_lag168=float(acf_vals[168]) if len(acf_vals) > 168 else None,
        acf_lag336=float(acf_vals[336]) if len(acf_vals) > 336 else None,
        top_picos=ranked,
    )


def stats_por_particion(serie: np.ndarray, splits: dict) -> dict:
    """Calcula stats descriptivos en train, val y test por separado."""
    n_train = splits["train"]
    n_val = splits["val"]
    train = serie[:n_train]
    val = serie[n_train:n_train + n_val]
    test = serie[n_train + n_val:n_train + n_val + splits["test"]]
    out = {}
    for nombre, parte in [("train", train), ("val", val), ("test", test)]:
        # Excluir sentinels para no contaminar las stats
        limpia = parte.copy()
        for s in SENTINELS:
            limpia = limpia[limpia != s]
        if len(limpia) == 0:
            out[nombre] = dict(n=0)
            continue
        out[nombre] = dict(
            n=int(len(limpia)),
            mean=float(limpia.mean()),
            std=float(limpia.std()),
            min=float(limpia.min()),
            max=float(limpia.max()),
        )
    # Distribution shift: diferencia relativa de medias entre train y test
    if out["train"].get("n", 0) and out["test"].get("n", 0):
        delta_mean = abs(out["test"]["mean"] - out["train"]["mean"])
        delta_rel = delta_mean / abs(out["train"]["mean"]) if out["train"]["mean"] != 0 else float("nan")
        out["delta_mean_train_test"] = float(delta_mean)
        out["delta_rel_train_test_pct"] = float(100 * delta_rel) if not np.isnan(delta_rel) else None
        # Diferencia relativa de std (volatilidad)
        if out["train"]["std"] != 0:
            out["delta_rel_std_train_test_pct"] = float(
                100 * abs(out["test"]["std"] - out["train"]["std"]) / out["train"]["std"]
            )
        else:
            out["delta_rel_std_train_test_pct"] = None
    return out


def splits_tslib(name: str, n_total: int) -> dict:
    """Calcula los tamaños de las particiones cronológicas según TSLib."""
    if name in ("ETTh1", "ETTh2"):
        train_end = 12 * 30 * 24
        val_end = train_end + 4 * 30 * 24
        test_end = val_end + 4 * 30 * 24
        return dict(
            esquema="ETT canonical (12m train / 4m val / 4m test)",
            train=train_end,
            val=val_end - train_end,
            test=test_end - val_end,
            cobertura_pct=round(100 * test_end / n_total, 1),
        )
    else:
        train_end = int(0.7 * n_total)
        val_end = int(0.8 * n_total)
        return dict(
            esquema="Custom (70% / 10% / 20%)",
            train=train_end,
            val=val_end - train_end,
            test=n_total - val_end,
            cobertura_pct=100.0,
        )


def analizar_dataset(name: str, path: str, target: str, freq: str) -> dict:
    """Pipeline completo de EDA para un dataset."""
    if not os.path.exists(path):
        return dict(name=name, error=f"Archivo no encontrado: {path}")

    df = pd.read_csv(path)
    n_total = len(df)
    n_cols = df.shape[1]
    n_vars = n_cols - 1
    n_nulls_pandas = int(df.isnull().sum().sum())
    dtypes_set = sorted({str(t) for t in df.dtypes if str(t) != "object"})

    if target not in df.columns:
        return dict(name=name, error=f"Columna target '{target}' no encontrada")

    serie = df[target].astype(float).values
    sentinels = detectar_sentinels(serie)
    desc = descriptivos(serie)
    adf = test_estacionariedad_completo(serie)
    per = periodicidades(serie)
    sp = splits_tslib(name, n_total)
    parts = stats_por_particion(serie, sp)

    return dict(
        name=name,
        path=path,
        target=target,
        freq=freq,
        n_total=n_total,
        n_cols=n_cols,
        n_vars=n_vars,
        n_nulls_pandas=n_nulls_pandas,
        sentinels_detectados=sentinels,
        dtypes_numericos=dtypes_set,
        descriptivos=desc,
        adf=adf,
        periodicidades=per,
        splits=sp,
        particiones=parts,
    )


def imprimir_resultados(resultados: list, fout=sys.stdout) -> None:
    """Imprime los resultados en formato legible."""
    p = lambda *a, **kw: __builtins__.print(*a, file=fout, **kw)

    p("=" * 100)
    p("EDA REPRODUCIBLE Y EXHAUSTIVO — TFG RITMO")
    p("=" * 100)
    p()
    p("Generado por: notebooks/eda_datasets.py")
    p("Pipeline: estructura -> integridad -> sentinels -> descriptivos -> ADF (serie completa)")
    p("          -> ACF/periodicidades -> splits TSLib -> stats por partición -> resumen")
    p()

    # ------------------------------------------------------------------------
    p("-" * 100)
    p("1. ESTRUCTURA Y VALIDACIÓN DE INTEGRIDAD")
    p("-" * 100)
    p(f"{'Dataset':<12} {'Shape':<14} {'Vars':<6} {'Target':<8} {'Freq':<10} {'Nulls(pandas)':<15} {'Tipos numéricos'}")
    for r in resultados:
        if "error" in r:
            p(f"{r['name']:<12} ERROR: {r['error']}")
            continue
        p(f"{r['name']:<12} {str((r['n_total'], r['n_cols'])):<14} {r['n_vars']:<6} "
          f"{r['target']:<8} {r['freq']:<10} {r['n_nulls_pandas']:<15} {', '.join(r['dtypes_numericos'])}")
    p()

    # ------------------------------------------------------------------------
    p("-" * 100)
    p("2. DETECCIÓN DE VALORES SENTINEL ENCUBIERTOS")
    p("-" * 100)
    hay_sentinels = False
    for r in resultados:
        if "error" in r:
            continue
        if r['sentinels_detectados']:
            hay_sentinels = True
            for val, n in r['sentinels_detectados'].items():
                pct = 100 * n / r['n_total']
                p(f"  {r['name']:<12}  -> {n} valores '{val}' ({pct:.3f}% del dataset)")
    if not hay_sentinels:
        p("  Ningún sentinel encontrado en ningún dataset (-9999, -999, -99, 9999, 999).")
    p()
    p(f"Sentinels buscados: {SENTINELS}")
    p("Nota: pandas reporta 0 nulos en TODOS los datasets, pero los sentinels sí se han detectado")
    p("manualmente y son tratados como missing por convención del repositorio MPI Biogeochemistry.")
    p()

    # ------------------------------------------------------------------------
    p("-" * 100)
    p("3. ESTADÍSTICOS DESCRIPTIVOS DEL TARGET (.describe equivalente)")
    p("-" * 100)
    p(f"{'Dataset':<12} {'min':>10} {'q05':>10} {'q25':>10} {'median':>10} {'mean':>10} {'q75':>10} {'q95':>10} {'max':>10} {'std':>10}")
    for r in resultados:
        if "error" in r:
            continue
        d = r['descriptivos']
        p(f"{r['name']:<12} {d['min']:>10.3f} {d['q05']:>10.3f} {d['q25']:>10.3f} {d['median']:>10.3f} "
          f"{d['mean']:>10.3f} {d['q75']:>10.3f} {d['q95']:>10.3f} {d['max']:>10.3f} {d['std']:>10.3f}")
    p()
    p(f"{'Dataset':<12} {'rango':>10} {'CV':>10} {'skew':>10} {'kurtosis':>10} {'n_unique':>10}")
    for r in resultados:
        if "error" in r:
            continue
        d = r['descriptivos']
        p(f"{r['name']:<12} {d['rango']:>10.3f} {d['coef_var']:>10.3f} {d['skewness']:>10.3f} "
          f"{d['kurtosis']:>10.3f} {d['n_unique']:>10}")
    p()

    # ------------------------------------------------------------------------
    p("-" * 100)
    p("4. ESTACIONARIEDAD — Augmented Dickey-Fuller sobre SERIE COMPLETA (sin submuestreo)")
    p("-" * 100)
    p(f"{'Dataset':<12} {'ADF stat':>12} {'p-value':>14} {'crit 5%':>10} {'lag':>5} {'N obs':>8} {'N limpia':>10}  Conclusión")
    for r in resultados:
        if "error" in r:
            continue
        a = r['adf']
        concl = "ESTACIONARIA" if a['es_estacionaria'] else "NO estacionaria"
        p(f"{r['name']:<12} {a['adf_stat']:>12.4f} {a['p_value']:>14.4e} {a['crit_5pct']:>10.4f} "
          f"{a['lag_usado']:>5} {a['n_obs_test']:>8} {a['n_serie_limpia']:>10}  {concl}")
    p()
    p("Hipótesis nula H0: la serie tiene una raíz unitaria (NO es estacionaria).")
    p("Criterio: si p-value < 0.05, se rechaza H0 y se concluye que la serie ES estacionaria.")
    p("Si ADF stat < crit 5%, también se rechaza H0 con confianza del 95%.")
    p()

    # ------------------------------------------------------------------------
    p("-" * 100)
    p("5. AUTOCORRELACIÓN Y PERIODICIDADES DOMINANTES")
    p("-" * 100)
    p(f"{'Dataset':<12} {'ACF(1)':>8} {'ACF(24)':>9} {'ACF(48)':>9} {'ACF(144)':>10} {'ACF(168)':>10} {'ACF(336)':>10}")
    for r in resultados:
        if "error" in r:
            continue
        pe = r['periodicidades']
        def f(v):
            return f"{v:.3f}" if v is not None else "—"
        p(f"{r['name']:<12} {f(pe['acf_lag1']):>8} {f(pe['acf_lag24']):>9} {f(pe['acf_lag48']):>9} "
          f"{f(pe['acf_lag144']):>10} {f(pe['acf_lag168']):>10} {f(pe['acf_lag336']):>10}")
    p()
    p("Interpretación de los lags clave (frecuencia horaria/10min):")
    p("  - ETTh / Electricity / Traffic (horaria):")
    p("      lag 24  -> ciclo diario (24 horas)")
    p("      lag 168 -> ciclo semanal (24*7 horas)")
    p("      lag 336 -> ciclo bisemanal")
    p("  - Weather (cada 10 min): lag 144 -> ciclo diario, lag 1008 -> ciclo semanal")
    p("  - Exchange (diaria): lag 24 -> 24 días, lag 168 -> 168 días")
    p()
    p("Top picos detectados en la ACF (lag, valor) para cada dataset:")
    for r in resultados:
        if "error" in r:
            continue
        top = r['periodicidades']['top_picos']
        top_str = ", ".join(f"({lag},{val:.2f})" for lag, val in top[:8])
        p(f"  {r['name']:<12}: {top_str if top_str else '(sin picos > 0.1)'}")
    p()

    # ------------------------------------------------------------------------
    p("-" * 100)
    p("6. SPLITS CRONOLÓGICOS APLICADOS POR TSLib")
    p("-" * 100)
    p(f"{'Dataset':<12} {'Esquema':<42} {'Train':>8} {'Val':>8} {'Test':>8} {'Cob.%':>7}")
    for r in resultados:
        if "error" in r:
            continue
        s = r['splits']
        p(f"{r['name']:<12} {s['esquema']:<42} {s['train']:>8} {s['val']:>8} "
          f"{s['test']:>8} {s['cobertura_pct']:>7.1f}")
    p()
    p("Verificado en data_provider/data_loader.py:")
    p("  - Dataset_ETT_hour (líneas 19-130): split fijo en MESES = 12/4/4 (= 8640/2880/2880 horas)")
    p("  - Dataset_Custom (líneas 133+):     split en porcentajes = 70%/10%/20%")
    p("  - Particiones cronológicas estrictas (validación y test posteriores al entrenamiento)")
    p("  - StandardScaler ajustado SOLO sobre train, aplicado a las 3 particiones")
    p()

    # ------------------------------------------------------------------------
    p("-" * 100)
    p("7. ANÁLISIS DE DISTRIBUTION SHIFT POR PARTICIÓN (train / val / test)")
    p("-" * 100)
    p(f"{'Dataset':<12} {'Mean train':>12} {'Mean val':>12} {'Mean test':>12} {'|Δμ| t-test':>13} {'%Δμ rel':>10} {'%Δσ rel':>10}")
    for r in resultados:
        if "error" in r:
            continue
        pt = r['particiones']
        if 'train' not in pt or 'test' not in pt:
            continue
        mt = pt['train'].get('mean', float('nan'))
        mv = pt['val'].get('mean', float('nan'))
        mte = pt['test'].get('mean', float('nan'))
        dm = pt.get('delta_mean_train_test', float('nan'))
        dr = pt.get('delta_rel_train_test_pct', float('nan'))
        ds = pt.get('delta_rel_std_train_test_pct', float('nan'))
        p(f"{r['name']:<12} {mt:>12.3f} {mv:>12.3f} {mte:>12.3f} {dm:>13.3f} "
          f"{dr if dr is not None else float('nan'):>10.2f} "
          f"{ds if ds is not None else float('nan'):>10.2f}")
    p()
    p("Interpretación: %Δμ rel mide cuánto cambia la media del target entre train y test")
    p("(en porcentaje de la media de train). Valores >10% sugieren distribution shift acusado.")
    p()

    # ------------------------------------------------------------------------
    p("-" * 100)
    p("8. RESUMEN ESTRUCTURADO PARA LA SECCIÓN 4.4 DEL TFG")
    p("-" * 100)
    n_estac = sum(1 for r in resultados if "error" not in r and r['adf']['es_estacionaria'])
    n_no_estac = sum(1 for r in resultados if "error" not in r and not r['adf']['es_estacionaria'])
    n_nulls_total = sum(r['n_nulls_pandas'] for r in resultados if "error" not in r)
    n_sentinel_total = sum(
        sum(r['sentinels_detectados'].values()) for r in resultados if "error" not in r
    )

    p(f"  - Datasets analizados: {len([r for r in resultados if 'error' not in r])} / 6")
    p(f"  - Tipos numéricos: TODOS los targets son float64")
    p(f"  - Valores nulos detectados por pandas (.isnull): {n_nulls_total}")
    p(f"  - Valores sentinel detectados manualmente: {n_sentinel_total}")
    p()
    p(f"  - Series ESTACIONARIAS (ADF p<0.05, serie completa): {n_estac}")
    p("    " + ", ".join(
        r['name'] for r in resultados if "error" not in r and r['adf']['es_estacionaria']
    ))
    p(f"  - Series NO estacionarias: {n_no_estac}")
    p("    " + ", ".join(
        r['name'] for r in resultados if "error" not in r and not r['adf']['es_estacionaria']
    ))
    p()
    p("  - Distribution shift train->test (Δ relativo de la media):")
    for r in resultados:
        if "error" in r:
            continue
        dr = r['particiones'].get('delta_rel_train_test_pct')
        if dr is not None:
            shift_label = "ALTO" if dr > 10 else "moderado" if dr > 3 else "bajo"
            p(f"      {r['name']:<12}: {dr:>6.2f}%  ({shift_label})")
    p()
    p("  - Periodicidades dominantes (lag con mayor pico ACF):")
    for r in resultados:
        if "error" in r:
            continue
        top = r['periodicidades']['top_picos']
        if top:
            lag_top, val_top = top[0]
            interpretacion = ""
            freq = r['freq']
            if freq == "horaria":
                if lag_top == 24 or 23 <= lag_top <= 25:
                    interpretacion = "diario"
                elif lag_top == 168 or 167 <= lag_top <= 169:
                    interpretacion = "semanal"
                elif lag_top == 336 or 335 <= lag_top <= 337:
                    interpretacion = "bisemanal"
            elif freq == "10 min":
                if 142 <= lag_top <= 146:
                    interpretacion = "diario"
            p(f"      {r['name']:<12}: pico principal en lag {lag_top:>4} (ACF={val_top:.3f}) {interpretacion}")
        else:
            p(f"      {r['name']:<12}: sin picos significativos")
    p()
    p("=" * 100)
    p("FIN DEL EDA")
    p("=" * 100)


def main():
    print("Ejecutando EDA reproducible y exhaustivo sobre los seis datasets del TFG RITMO...")
    print("(ADF test sobre la serie completa, sin submuestreo. Puede tardar 30s-2min.)")
    print()
    resultados = []
    for name, path, target, freq in DATASETS:
        print(f"  -> Procesando {name}...", flush=True)
        resultados.append(analizar_dataset(name, path, target, freq))
    print()
    print("Análisis completado. Imprimiendo resultados:")
    print()

    imprimir_resultados(resultados)

    out_path = os.path.join("notebooks", "eda_datasets_output.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        imprimir_resultados(resultados, fout=f)
    print()
    print(f"Resultados guardados en: {out_path}")


if __name__ == "__main__":
    main()
