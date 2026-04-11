"""
Figura 4.9. Diagrama del flujo de integracion del TransformerCommon
con las seis tecnicas de tokenizacion del Plan A.

Estilo: paper-quality (NeurIPS / ICML / ICLR), generado con Graphviz.
- Layout limpio sin solapamientos: el cluster del tokenizer recibe
  una unica flecha de entrada y emite una unica flecha de salida.
- HMM cache aislado en su propia franja inferior, conectado mediante
  un puerto invisible al tokenizer HMM.
- Backward gradient con constraint=false (no distorsiona el layout).
- Paleta Okabe-Ito (colorblind-safe) consistente con el TFG.
"""
from graphviz import Digraph

g = Digraph("ritmo_pipeline", format="png")

# === Atributos globales ===
g.attr(
    rankdir="LR",
    splines="spline",
    nodesep="0.45",
    ranksep="1.20",
    fontname="Helvetica",
    fontsize="13",
    bgcolor="white",
    pad="0.5",
    margin="0",
    concentrate="false",
    newrank="true",
    dpi="220",
)
g.attr("node",
       shape="box",
       style="rounded,filled",
       fontname="Helvetica",
       fontsize="12",
       penwidth="1.6",
       height="0.85",
       width="1.65",
       margin="0.18,0.14",
)
g.attr("edge",
       color="#3D3D3D",
       arrowsize="0.85",
       penwidth="1.5",
       fontname="Helvetica-Oblique",
       fontsize="11",
)

# === Paleta Okabe-Ito ===
COLOR_INPUT_FC = "#E8F4F8"
COLOR_INPUT_BC = "#0072B2"
COLOR_REVIN_FC = "#FFF4E0"
COLOR_REVIN_BC = "#E69F00"
COLOR_EMB_FC = "#E8F5E0"
COLOR_EMB_BC = "#009E73"
COLOR_TX_FC = "#D6E5F5"
COLOR_TX_BC = "#0072B2"
COLOR_HMM_FC = "#F5E1E0"
COLOR_HMM_BC = "#C0392B"
COLOR_FROZEN = "#C0392B"
COLOR_GRAD = "#0072B2"

# === Nodos principales (forward chain) ===
g.node("loader",
       "<<B>Data loader</B><BR/><FONT POINT-SIZE='10'>x &#8712; &#8477;<SUP>[B,96,1]</SUP></FONT>>",
       fillcolor=COLOR_INPUT_FC, color=COLOR_INPUT_BC)

g.node("revin",
       "<<B>RevIN</B><BR/>normalize<BR/><FONT POINT-SIZE='10'>(per-window)</FONT>>",
       fillcolor=COLOR_REVIN_FC, color=COLOR_REVIN_BC)

g.node("emb",
       "<<B>Embedding</B><BR/><B>generator</B>>",
       fillcolor=COLOR_EMB_FC, color=COLOR_EMB_BC)

g.node("tfm",
       "<<B>TransformerCommon</B><BR/><FONT POINT-SIZE='10'>encoder + flatten<BR/>+ linear head</FONT>>",
       fillcolor=COLOR_TX_FC, color=COLOR_TX_BC, penwidth="2.8",
       width="2.30", height="1.10")

g.node("rinv",
       "<<B>RevIN<SUP>-1</SUP></B><BR/>denormalize>",
       fillcolor=COLOR_REVIN_FC, color=COLOR_REVIN_BC)

g.node("yhat",
       "<<B>&#375;</B>  <FONT POINT-SIZE='10'>(escala original)</FONT>>",
       shape="note", fillcolor="white", color="#2C3E50",
       style="filled", penwidth="1.5")

# === Cluster del tokenizer (6 tecnicas) con un punto de entrada/salida ===
with g.subgraph(name="cluster_tok") as c:
    c.attr(
        label=("<<B><FONT POINT-SIZE='12' COLOR='#555555'>"
               "Tokenizer</FONT></B>"
               "<FONT POINT-SIZE='10' COLOR='#888888'> (sec. 4.2.7)</FONT>>"),
        style="dashed,rounded",
        color="#999999",
        labeljust="c",
        labelloc="t",
        margin="20",
        bgcolor="#FBFBFB",
        penwidth="1.2",
    )
    c.attr("node",
           shape="box",
           style="rounded,filled",
           fontname="Helvetica",
           fontsize="11",
           penwidth="1.4",
           height="0.55",
           width="2.10",
           margin="0.14,0.08",
    )
    # Punto de entrada invisible para fan-in
    c.node("tok_in", "", shape="point", width="0.01",
           style="invis")
    c.node("tok_out", "", shape="point", width="0.01",
           style="invis")

    # Orden visual: SAX arriba, HMM abajo
    c.node("t_hmm",   "<<B>HMM  ·  RITMO</B>>",
           fillcolor="#FFE4E1", color="#C0392B", penwidth="1.7")
    c.node("t_fnd",   "Foundation  ·  MOMENT",
           fillcolor="#E0F0FF", color="#0072B2")
    c.node("t_dec",   "Descomposicion  ·  Autoformer",
           fillcolor="#E0F0FF", color="#0072B2")
    c.node("t_patch", "Patching  ·  PatchTST",
           fillcolor="#E0F0FF", color="#0072B2")
    c.node("t_text",  "Text-based  ·  LLMTime",
           fillcolor="#FFEFD5", color="#E69F00")
    c.node("t_disc",  "Discretizacion  ·  SAX",
           fillcolor="#FFEFD5", color="#E69F00")

    # Edges invisibles para fan-in/fan-out (fuerzan layout vertical de las 6 tecnicas)
    for tid in ["t_disc", "t_text", "t_patch", "t_dec", "t_fnd", "t_hmm"]:
        c.edge("tok_in", tid, style="invis")
        c.edge(tid, "tok_out", style="invis")

# === HMM cache (frozen, lateral abajo) ===
# Altura mas grande para que el contenido respire dentro del cilindro
g.node("hmm_cache",
       ("<<B>HMM cache</B>"
        "<BR/> "
        "<BR/><FONT POINT-SIZE='10'>{ A,  &#956;<SUB>k</SUB>,  &#963;<SUB>k</SUB>,  &#960; }</FONT>"
        "<BR/> "
        "<BR/><FONT POINT-SIZE='10' COLOR='#C0392B'><I>frozen</I></FONT>>"),
       shape="cylinder",
       style="filled",
       fillcolor=COLOR_HMM_FC,
       color=COLOR_HMM_BC,
       penwidth="1.5",
       height="2.00",
       width="2.00",
       fontname="Helvetica",
       fontsize="11",
)

# === Forward edges principales (linea unica visible) ===
g.edge("loader", "revin", label=" x [B,96,1] ", fontcolor="#444444")
g.edge("revin", "tok_in", label=" x_norm ", fontcolor="#444444",
       arrowhead="none")
g.edge("tok_out", "emb", label=" tokens ", fontcolor="#444444")
g.edge("emb", "tfm", label=" [B,seq,d_m] ", fontcolor="#444444")
g.edge("tfm", "rinv", label=" [B,O,1] ", fontcolor="#444444")
g.edge("rinv", "yhat", label=" &#375;&#183;&#963;+&#956; ", fontcolor="#444444")

# === HMM cache -> tokenizer HMM (frozen, dashed red, una sola flecha limpia) ===
# Solo dibujamos la conexion HMM cache -> HMM tokenizer.
# Los parametros HMM (mu_k, sigma_k, A) llegan al embedding generator
# de forma natural a traves del tokenizer HMM, asi que una sola flecha
# representa toda la cadena de dependencias frozen sin solapamientos.
g.edge("hmm_cache:e", "t_hmm:w",
       label=" params HMM\n(frozen) ",
       style="dashed",
       color=COLOR_FROZEN,
       fontcolor=COLOR_FROZEN,
       arrowhead="open",
       penwidth="1.5",
       constraint="false")

# === Backward gradient (Transformer -> Embedding, dashed blue) ===
g.edge("tfm", "emb",
       label=" gradients (backprop) ",
       style="dashed",
       color=COLOR_GRAD,
       fontcolor=COLOR_GRAD,
       arrowhead="vee",
       penwidth="1.3",
       dir="back",
       constraint="false")

# === Forzar HMM cache abajo (rank sink) ===
with g.subgraph() as s:
    s.attr(rank="sink")
    s.node("hmm_cache")

# === Render: PNG (raster), PDF y SVG (vectoriales) ===
out_base = "/home/jaime/TFG/RITMO/notebooks/figures/figura_4_9_pipeline_transformer"
for fmt in ("png", "pdf", "svg"):
    g.format = fmt
    g.render(filename=out_base, cleanup=True, format=fmt)
    print(f"Generado: {out_base}.{fmt}")
