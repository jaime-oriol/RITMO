"""
Regenera todas las figuras SVG del TFG con maxima calidad vectorial.

Problemas corregidos:
1. imshow() embede PNG dentro del SVG (rasterizado). Se reemplaza por pcolormesh
   con edgecolors='face' que es 100% vectorial.
2. Fuentes exportadas como paths (no dependen de fuentes del sistema) con
   svg.fonttype='path' para consistencia de renderizado.
3. DPI=300 explicito para PDF (aunque SVG es resolucion-agnostico).
4. metadata limpia y bbox_inches='tight' para recorte preciso.

Estrategia: parchea notebooks in-memory (sin modificar disco) y ejecuta todas
las celdas con nbclient. Regenera los 23 SVGs desde cero.
"""
import json
import re
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path("/home/jaime/TFG/RITMO")
NOTEBOOKS = [
    "notebooks/visualizations.ipynb",
    "notebooks/eda_datasets.ipynb",
    "notebooks/pipeline_RITMO_etth2.ipynb",
    "tecnicas/ETTh2_tokenization.ipynb",
    "tecnicas/comparacion_metricas.ipynb",
]

# Configuracion de maxima calidad para SVG + monkey-patch de imshow
# El monkey-patch reemplaza ax.imshow() en runtime por pcolormesh vectorial,
# manteniendo las coordenadas de ax.text() compatibles.
RC_PARAMS_CODE = r"""
import matplotlib
import matplotlib.pyplot as _plt
import numpy as _np

# rcParams para maxima calidad vectorial
matplotlib.rcParams['svg.fonttype'] = 'path'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.hashsalt'] = 'ritmo'
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.transparent'] = False
matplotlib.rcParams['path.simplify'] = False

# Monkey-patch Axes.imshow para producir SVG vectorial
from matplotlib.axes import Axes
_orig_imshow = Axes.imshow

def _vector_imshow(self, X, *args, **kwargs):
    # Extraer parametros compatibles con pcolormesh
    cmap = kwargs.pop('cmap', None)
    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    norm = kwargs.pop('norm', None)
    alpha = kwargs.pop('alpha', None)
    aspect = kwargs.pop('aspect', None)
    # Ignorar parametros incompatibles
    kwargs.pop('interpolation', None)
    kwargs.pop('origin', None)
    kwargs.pop('extent', None)
    kwargs.pop('resample', None)
    kwargs.pop('filternorm', None)
    kwargs.pop('filterrad', None)
    kwargs.pop('interpolation_stage', None)

    arr = _np.asarray(X)
    if arr.ndim != 2:
        # RGBA o 3D: fallback a imshow original (no se puede vectorizar facilmente)
        kwargs.update(dict(cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, alpha=alpha))
        if aspect is not None:
            kwargs['aspect'] = aspect
        return _orig_imshow(self, X, *args, **kwargs)

    H, W = arr.shape
    # Edges centrados para que celda (i,j) este en (j, i)
    x_edges = _np.arange(W + 1) - 0.5
    y_edges = _np.arange(H + 1) - 0.5

    mesh = self.pcolormesh(
        x_edges, y_edges, arr,
        cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, alpha=alpha,
        edgecolors='face', linewidth=0, shading='flat', rasterized=False,
    )
    # Invertir eje Y para simular imshow top-left origin
    if not self.yaxis_inverted():
        self.invert_yaxis()
    # Mantener aspect ratio similar
    if aspect == 'equal' or aspect is None:
        self.set_aspect('equal')
    else:
        self.set_aspect('auto')
    # Limitar ejes al rango de la matriz (como imshow)
    self.set_xlim(x_edges[0], x_edges[-1])
    self.set_ylim(y_edges[-1], y_edges[0])  # invertido
    return mesh

Axes.imshow = _vector_imshow

# Monkey-patch Figure.colorbar para desactivar rasterization
from matplotlib.figure import Figure
_orig_colorbar = Figure.colorbar

def _vector_colorbar(self, mappable=None, cax=None, ax=None, **kwargs):
    cbar = _orig_colorbar(self, mappable=mappable, cax=cax, ax=ax, **kwargs)
    try:
        cbar.solids.set_rasterized(False)
        cbar.solids.set_edgecolor('face')
    except Exception:
        pass
    return cbar

Figure.colorbar = _vector_colorbar

# Tambien plt.colorbar
_orig_plt_colorbar = _plt.colorbar
def _vector_plt_colorbar(*args, **kwargs):
    cbar = _orig_plt_colorbar(*args, **kwargs)
    try:
        cbar.solids.set_rasterized(False)
        cbar.solids.set_edgecolor('face')
    except Exception:
        pass
    return cbar
_plt.colorbar = _vector_plt_colorbar
"""


def find_matching_paren(s: str, start: int) -> int:
    """Dado s[start]='(', devuelve indice del ')' que cierra."""
    assert s[start] == '('
    depth = 0
    i = start
    while i < len(s):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def patch_colorbar_rasterized(src: str) -> str:
    """Anade cbar.solids.set_rasterized(False) tras cada llamada a colorbar().

    Por defecto matplotlib rasteriza los colorbars en SVG como PNG embebido.
    Desactivamos la rasterizacion explicitamente.
    """
    pattern = re.compile(
        r'^(\s*)(\w+)\s*=\s*(?:plt|fig|figure)\.colorbar\([^)]*\).*$',
        re.MULTILINE
    )
    def insert_fix(m):
        indent = m.group(1)
        var = m.group(2)
        return f"{m.group(0)}\n{indent}{var}.solids.set_rasterized(False)"
    result = pattern.sub(insert_fix, src)

    # Tambien manejar plt.colorbar(...) sin asignacion
    # -> transformar a _cbar = plt.colorbar(...); _cbar.solids.set_rasterized(False)
    pattern2 = re.compile(
        r'^(\s*)(plt|fig|figure)\.colorbar\(([^)]*)\)\s*$',
        re.MULTILINE
    )
    def insert_fix2(m):
        indent = m.group(1)
        obj = m.group(2)
        args = m.group(3)
        return f"{indent}_cbar_tmp = {obj}.colorbar({args})\n{indent}_cbar_tmp.solids.set_rasterized(False)"
    result = pattern2.sub(insert_fix2, result)

    return result


def patch_imshow_to_pcolormesh(src: str) -> str:
    """Reemplaza imshow() por pcolormesh() equivalente 100% vectorial.

    Estrategia: busca manualmente cada llamada a imshow y balancea parentesis
    correctamente para extraer los argumentos completos (aunque contengan
    llamadas anidadas como len(COL_ORDER)).
    """
    if 'imshow' not in src:
        return src

    # Encontrar todas las llamadas .imshow(
    pattern = re.compile(r'(\w+)\.imshow\(')

    result = []
    last_end = 0
    # matches list para procesar en orden inverso (no romper indices)
    matches = list(pattern.finditer(src))

    for m in matches:
        obj = m.group(1)
        open_paren = m.end() - 1  # posicion del '('
        close_paren = find_matching_paren(src, open_paren)
        if close_paren == -1:
            continue

        # Argumentos entre parentesis (sin los parentesis mismos)
        args = src[open_paren + 1 : close_paren]

        # Quitar aspect, interpolation, origin que no tiene pcolormesh
        args = re.sub(r",?\s*aspect\s*=\s*['\"][^'\"]+['\"]", '', args)
        args = re.sub(r",?\s*interpolation\s*=\s*['\"][^'\"]+['\"]", '', args)
        args = re.sub(r",?\s*origin\s*=\s*['\"][^'\"]+['\"]", '', args)

        # Limpiar comas/espacios finales
        args = args.rstrip(', \n\t')

        # Anadir parametros vectoriales
        new_args = args + ", edgecolors='face', linewidth=0, shading='auto', rasterized=False"

        # Construir reemplazo: copiar prefijo + pcolormesh + nuevos args + )
        result.append(src[last_end : m.start()])
        result.append(f"{obj}.pcolormesh({new_args})")
        last_end = close_paren + 1

    result.append(src[last_end:])
    new_src = ''.join(result)

    # pcolormesh necesita y-axis invertido (imshow lo tiene por defecto)
    # y set_aspect('auto') para comportamiento similar a imshow(aspect='auto').
    # Estrategia: buscar la llamada pcolormesh completa y el cierre ')' (balanceado),
    # luego anadir invert_yaxis tras la SIGUIENTE newline.
    pcolor_pattern = re.compile(r'(\w+)\.pcolormesh\(')
    pcolor_matches = list(pcolor_pattern.finditer(new_src))
    # Procesar en orden inverso para no romper indices
    for pm in reversed(pcolor_matches):
        obj_left = pm.group(1)  # puede ser 'im' o 'ax' directamente
        open_p = pm.end() - 1
        close_p = find_matching_paren(new_src, open_p)
        if close_p == -1:
            continue
        # Encontrar fin de la linea que contiene close_p
        nl = new_src.find('\n', close_p)
        if nl == -1:
            nl = len(new_src)

        # Determinar indent de la linea que contiene la llamada
        line_start = new_src.rfind('\n', 0, pm.start()) + 1
        indent = ''
        for ch in new_src[line_start:pm.start()]:
            if ch in ' \t':
                indent += ch
            else:
                break

        # Determinar el 'ax' real. Si pm.group(1) es 'im' o similar, buscar
        # el ax en el prefijo 'im = ax'
        ax_name = obj_left
        prefix = new_src[line_start : pm.start()]
        m_eq = re.search(r'=\s*(\w+)\s*$', prefix.rstrip())
        if m_eq:
            ax_name = m_eq.group(1)

        # Solo anadir si es un objeto 'ax'-like (no 'plt', no 'fig')
        if ax_name in ('plt', 'fig', 'figure'):
            continue

        insertion = f"\n{indent}{ax_name}.invert_yaxis()\n{indent}{ax_name}.set_aspect('auto')"
        new_src = new_src[:nl] + insertion + new_src[nl:]
    return new_src


def inject_rcparams(nb: nbformat.NotebookNode) -> None:
    """Anade celda de rcParams al principio del notebook (en memoria)."""
    rc_cell = nbformat.v4.new_code_cell(source=RC_PARAMS_CODE)
    nb.cells.insert(0, rc_cell)


def process_notebook(nb_rel: str) -> dict:
    """Parchea y ejecuta un notebook. Devuelve stats."""
    nb_path = ROOT / nb_rel
    print(f"\n{'='*60}")
    print(f"  {nb_rel}")
    print(f"{'='*60}")

    if not nb_path.exists():
        print(f"  SKIP: no existe")
        return {'ok': False}

    nb = nbformat.read(nb_path, as_version=4)

    # Con monkey-patch en RC_PARAMS_CODE no hace falta modificar celdas.
    # imshow y colorbar se interceptan en runtime.
    n_patched = 0

    # Inyectar rcParams al principio
    inject_rcparams(nb)

    # Ejecutar notebook CON SU PROPIO directorio como cwd
    # (los notebooks usan paths relativos segun donde estan ubicados)
    nb_dir = nb_path.parent
    try:
        print(f"  executing notebook (cwd={nb_dir.name})...")
        client = NotebookClient(
            nb,
            timeout=600,
            kernel_name='python3',
            resources={'metadata': {'path': str(nb_dir)}}
        )
        client.execute()
        print(f"  OK ({n_patched} cells patched)")
        return {'ok': True, 'patched': n_patched}
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {str(e)[:300]}")
        return {'ok': False, 'error': str(e)[:300]}


if __name__ == "__main__":
    results = {}
    for nb in NOTEBOOKS:
        results[nb] = process_notebook(nb)

    print(f"\n{'='*60}")
    print("  RESUMEN")
    print(f"{'='*60}")
    for nb, r in results.items():
        status = "OK" if r.get('ok') else "FAIL"
        patched = r.get('patched', 0)
        print(f"  {status} {nb}: {patched} imshow patched")
