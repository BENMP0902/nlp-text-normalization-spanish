"""
visualizacion.py — Reducción de dimensionalidad y gráfico del espacio vectorial.
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib
matplotlib.use("Agg")   # backend sin ventana: obligatorio en WSL2 sin servidor X.
                        # Debe ir ANTES de importar pyplot o no surte efecto.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD


def proyectar(X, n_componentes: int = 2, semilla: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Proyecta la matriz documento-término a n_componentes dimensiones.

    TruncatedSVD y no PCA, por tres razones:
      1. PCA centra los datos → densifica la matriz → .toarray() → MemoryError
         en cualquier corpus real. TruncatedSVD opera sobre sparse directamente.
      2. PCA sobre conteos crudos produce una PC1 que es "longitud del documento".
      3. TruncatedSVD sobre TF-IDF **es** LSA (Deerwester et al., 1990). Las
         componentes latentes agrupan términos co-ocurrentes, así que sinónimos
         dejan de ser ortogonales. Esto es lo único aquí que merece el
         adjetivo "semántico".
    """
    svd = TruncatedSVD(n_components=n_componentes, random_state=semilla)
    coords = svd.fit_transform(X)
    return coords, svd.explained_variance_ratio_


def graficar(coords, etiquetas, varianza, titulo: str, ruta: Path) -> None:
    """
    Scatter 2D con varianza explicada EN EL TÍTULO.

    Reportar explained_variance_ratio_ no es burocracia: si el gráfico
    representa el 12% de la información, cualquier cluster que "veas" puede ser
    artefacto de proyección. El número es la defensa contra tu sistema visual.
    """
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for clase in sorted(set(etiquetas)):
        m = [e == clase for e in etiquetas]                  # máscara booleana por clase
        ax.scatter(coords[m, 0], coords[m, 1], s=70, alpha=0.8,
                   edgecolors="k", linewidths=0.5, label=clase)
    ax.axhline(0, color="grey", lw=0.5, ls="--")             # ejes de referencia en el origen
    ax.axvline(0, color="grey", lw=0.5, ls="--")
    ax.set_xlabel(f"Componente 1 ({varianza[0]:.1%})")       # varianza POR EJE
    ax.set_ylabel(f"Componente 2 ({varianza[1]:.1%})")
    ax.set_title(f"{titulo}\nVarianza acumulada: {varianza.sum():.1%}")
    ax.legend(title="Corpus")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)                                            # libera memoria: sin esto,
                                                              # N figuras abiertas = fuga
