"""
Parchea automaticamente todos los savefig(...) de los notebooks
para que ademas del PNG generen tambien PDF y SVG vectoriales.

Estrategia: tras cada llamada savefig(...png...) inyecta dos lineas
adicionales con .pdf y .svg, manteniendo todos los kwargs originales.

Esto convierte cualquier figura matplotlib en formato apto para Word
sin perder calidad al escalar.
"""
import json
import re
from pathlib import Path

NOTEBOOKS = [
    "tecnicas/ETTh2_tokenization.ipynb",
    "tecnicas/comparacion_metricas.ipynb",
    "notebooks/visualizations.ipynb",
    "notebooks/eda_datasets.ipynb",
    "notebooks/pipeline_RITMO_etth2.ipynb",
]

ROOT = Path("/home/jaime/TFG/RITMO")

# Patron: captura savefig(...) con su contenido completo
# Soporta: plt.savefig, fig.savefig, ax.savefig, etc.
SAVEFIG_RE = re.compile(
    r"((?:plt|fig|figure|ax|f)\.savefig\(\s*['\"])([^'\"]+\.png)(['\"][^)]*\))"
)


def patch_source(source: str) -> tuple[str, int]:
    """Anade savefig PDF/SVG despues de cada savefig PNG."""
    lines = source.split("\n") if isinstance(source, str) else source
    if isinstance(lines, list) and lines and not isinstance(lines[0], str):
        return source, 0

    out = []
    n_patched = 0
    for line in lines:
        out.append(line)
        match = SAVEFIG_RE.search(line)
        if match:
            # Comprobar si la siguiente linea ya tiene el patch (idempotente)
            already_patched = False
            # Generar la version PDF y SVG en las mismas condiciones
            prefix, png_path, suffix = match.group(1), match.group(2), match.group(3)
            base = png_path.rsplit(".png", 1)[0]
            pdf_line = line.replace(png_path, base + ".pdf")
            svg_line = line.replace(png_path, base + ".svg")
            # Marcamos cada patched line con un comentario invisible para idempotencia
            tag = "  # auto-vectorial"
            if pdf_line.strip().endswith(tag.strip()):
                already_patched = True
            if not already_patched:
                out.append(pdf_line + tag)
                out.append(svg_line + tag)
                n_patched += 1
    return "\n".join(out) if isinstance(source, str) else out, n_patched


def patch_notebook(nb_path: Path) -> int:
    with open(nb_path) as f:
        nb = json.load(f)

    total_patched = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        if "savefig" not in src or ".png" not in src:
            continue
        new_src, n = patch_source(src)
        if n > 0:
            cell["source"] = new_src.split("\n")
            # nbformat espera una lista con \n al final de cada linea excepto la ultima
            cell["source"] = [
                ln + "\n" for ln in cell["source"][:-1]
            ] + [cell["source"][-1]]
            total_patched += n

    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    return total_patched


if __name__ == "__main__":
    grand = 0
    for nb_rel in NOTEBOOKS:
        nb_path = ROOT / nb_rel
        if not nb_path.exists():
            print(f"  SKIP (no existe): {nb_rel}")
            continue
        n = patch_notebook(nb_path)
        print(f"  {nb_rel}: +{n} savefig parcheadas")
        grand += n
    print(f"\nTOTAL: {grand} savefig calls ahora generan PNG + PDF + SVG")
