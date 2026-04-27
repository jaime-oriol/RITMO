"""Wrapper resumible para todos los .sh de baselines SOTA (Plan B).

Parsea cada `python -u run.py ...` invocation de los .sh en
scripts/long_term_forecast/{ECL,ETT,Exchange,Traffic,Weather}_script/, construye el
`setting` que generaria run.py, comprueba si results/{setting}_0/metrics.npy ya existe
y solo ejecuta las pendientes.

Resumible: si lo paras y re-arrancas, salta las completadas.

Uso:
    python -u scripts/long_term_forecast/run_all_baselines.py
    python -u scripts/long_term_forecast/run_all_baselines.py --only-model PatchTST
    python -u scripts/long_term_forecast/run_all_baselines.py --only-dataset ETTh1
    python -u scripts/long_term_forecast/run_all_baselines.py --dry-run
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)

# Defaults de run.py (segun argparse en run.py).
RUN_PY_DEFAULTS = {
    'task_name': 'long_term_forecast',
    'features': 'M',
    'seq_len': '96',
    'label_len': '48',
    'd_model': '512',
    'n_heads': '8',
    'e_layers': '2',
    'd_layers': '1',
    'd_ff': '2048',
    'expand': '2',
    'd_conv': '4',
    'factor': '1',
    'embed': 'timeF',
    'distil': 'True',
    'des': 'test',
}

SHELL_DIRS = [
    REPO_ROOT / 'scripts/long_term_forecast/ECL_script',
    REPO_ROOT / 'scripts/long_term_forecast/ETT_script',
    REPO_ROOT / 'scripts/long_term_forecast/Exchange_script',
    REPO_ROOT / 'scripts/long_term_forecast/Traffic_script',
    REPO_ROOT / 'scripts/long_term_forecast/Weather_script',
]


def parse_run_py_invocations(sh_path):
    """Extrae cada bloque `python -u run.py ... \\` de un .sh y devuelve lista de dicts {flag: val}.

    Maneja sustitucion de variables bash (`$var` o `${var}`) extraidas del propio .sh.
    """
    text = sh_path.read_text()
    # Recolecta asignaciones bash: `var=value` (linea simple, valor sin espacios)
    bash_vars = {}
    for m in re.finditer(r'^(\w+)=([^\s#]+)\s*$', text, flags=re.MULTILINE):
        bash_vars[m.group(1)] = m.group(2)

    def expand(s):
        # Reemplaza $var y ${var}
        s = re.sub(r'\$\{(\w+)\}', lambda m: bash_vars.get(m.group(1), m.group(0)), s)
        s = re.sub(r'\$(\w+)', lambda m: bash_vars.get(m.group(1), m.group(0)), s)
        return s

    # Une lineas con backslash continuation
    joined = re.sub(r'\\\n\s*', ' ', text)
    invocations = []
    for line in joined.splitlines():
        line = line.strip()
        if not line.startswith('python ') or 'run.py' not in line:
            continue
        line = expand(line)  # expandir bash vars antes de tokenizar
        try:
            tokens = shlex.split(line)
        except ValueError:
            print(f'  WARN: shlex falla en {sh_path.name}: {line[:80]}')
            continue
        args = {}
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.startswith('--'):
                flag = tok[2:]
                if i + 1 < len(tokens) and not tokens[i+1].startswith('--'):
                    args[flag] = tokens[i+1]
                    i += 2
                else:
                    args[flag] = 'True'
                    i += 1
            else:
                i += 1
        if args:
            invocations.append(args)
    return invocations


def build_setting(args):
    """Construye el string `setting` igual que run.py:213."""
    g = lambda k: args.get(k, RUN_PY_DEFAULTS.get(k, ''))
    setting = (
        f"{g('task_name')}_{g('model_id')}_{g('model')}_{g('data')}"
        f"_ft{g('features')}_sl{g('seq_len')}_ll{g('label_len')}_pl{g('pred_len')}"
        f"_dm{g('d_model')}_nh{g('n_heads')}_el{g('e_layers')}_dl{g('d_layers')}_df{g('d_ff')}"
        f"_expand{g('expand')}_dc{g('d_conv')}_fc{g('factor')}_eb{g('embed')}_dt{g('distil')}"
        f"_{g('des')}_0"
    )
    return setting


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only-model', default=None, help='PatchTST | DLinear | TimeMixer | TimeXer')
    ap.add_argument('--only-dataset', default=None,
                    help='ETTh1 | ETTh2 | Weather | ECL | Traffic | Exchange (matchea data_arg)')
    ap.add_argument('--only-pred-len', type=int, default=None)
    ap.add_argument('--skip-cross-domain', action='store_true',
                    help='excluye Traffic y Exchange (mantiene solo in-domain Plan B: ETTh1/ETTh2/Weather/ECL)')
    ap.add_argument('--dry-run', action='store_true', help='solo imprime, no ejecuta')
    return ap.parse_args()


def main():
    args = parse_args()

    # Recolectar invocaciones de todos los .sh
    all_invocations = []
    for sh_dir in SHELL_DIRS:
        for sh_path in sorted(sh_dir.glob('*.sh')):
            invs = parse_run_py_invocations(sh_path)
            for inv in invs:
                inv['_source'] = str(sh_path.relative_to(REPO_ROOT))
                all_invocations.append(inv)

    print(f'[baselines] Cargadas {len(all_invocations)} invocaciones de '
          f'{sum(1 for d in SHELL_DIRS for _ in d.glob("*.sh"))} archivos .sh')

    # Filtrar
    if args.only_model:
        all_invocations = [inv for inv in all_invocations if inv.get('model') == args.only_model]
    if args.only_dataset:
        all_invocations = [inv for inv in all_invocations if inv.get('data') == args.only_dataset]
    if args.only_pred_len:
        all_invocations = [inv for inv in all_invocations if inv.get('pred_len') == str(args.only_pred_len)]
    if args.skip_cross_domain:
        # Excluye scripts que viven en Traffic_script o Exchange_script (cross-domain)
        all_invocations = [
            inv for inv in all_invocations
            if 'Traffic_script' not in inv.get('_source', '')
            and 'Exchange_script' not in inv.get('_source', '')
        ]

    print(f'[baselines] Tras filtros: {len(all_invocations)} experimentos a procesar\n')

    done = ok = skipped = failed = 0
    t_all = time.time()
    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '4')
    env.setdefault('MKL_NUM_THREADS', '4')

    # Sanity check: torch importable en el python actual
    try:
        import torch  # noqa: F401
        print(f'[baselines] torch OK en {sys.executable}', flush=True)
    except ImportError:
        print(f'\n[baselines] ERROR: torch NO encontrado en {sys.executable}\n'
              f'  Activa conda primero:\n'
              f'  source ~/anaconda3/etc/profile.d/conda.sh && conda activate ritmo\n'
              f'  Luego re-lanza este script.', flush=True)
        return

    for inv in all_invocations:
        done += 1
        setting = build_setting(inv)
        result_path = REPO_ROOT / 'results' / setting / 'metrics.npy'
        tag = (f"{inv.get('model','?')} {inv.get('data','?')} "
               f"pl={inv.get('pred_len','?')} ({inv.pop('_source')})")

        if result_path.exists():
            skipped += 1
            print(f'[{done}/{len(all_invocations)}] SKIP {tag}', flush=True)
            continue

        # Reconstruir cmd
        cmd = [sys.executable, '-u', 'run.py']
        for k, v in inv.items():
            if v == 'True':
                cmd.append(f'--{k}')
            else:
                cmd += [f'--{k}', str(v)]

        if args.dry_run:
            print(f'[{done}/{len(all_invocations)}] DRY  {tag}')
            continue

        print(f'\n{"="*80}\n[{done}/{len(all_invocations)}] RUN  {tag}\n{"="*80}', flush=True)
        t0 = time.time()
        # stdout/stderr fluyen directamente al padre (que esta capturado por nohup en el log).
        proc = subprocess.run(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
        dt = time.time() - t0

        if proc.returncode != 0:
            failed += 1
            print(f'\n[{done}/{len(all_invocations)}] FAIL ({dt:.0f}s)\n', flush=True)
        else:
            ok += 1
            print(f'\n[{done}/{len(all_invocations)}] OK   ({dt:.0f}s)\n', flush=True)

    print(f'\n[baselines] Fin. ok={ok} skipped={skipped} failed={failed} '
          f'total={done}  wall={time.time()-t_all:.1f}s')


if __name__ == '__main__':
    main()
