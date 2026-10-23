"""
visualizacion.py — Reducción de dimensionalidad y gráfico del espacio vectorial.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import matplotlib
matplotlib.use("Agg")   # Backend sin ventana. OBLIGATORIO en WSL2 sin servidor X.
                        # Debe ir ANTES de importar pyplot o no surte efecto.
import matplotlib.pyplot as plt          # noqa: E402  (import tras matplotlib.use, a propósito)
import numpy as np                        # noqa: E402
from sklearn.decomposition import PCA, TruncatedSVD   # noqa: E402


def proyectar(X, n_componentes: int = 2, semilla: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Proyecta la matriz documento-término a n_componentes dimensiones.

    TruncatedSVD y no PCA, por tres razones acumulativas:
      1. PCA centra los datos restando la media → densifica la matriz dispersa →
         obliga a .toarray() → MemoryError en cualquier corpus real. La
         documentación de scikit-learn recomienda TruncatedSVD explícitamente
         para matrices término-documento.
      2. PCA sobre conteos crudos produce una PC1 dominada por la LONGITUD del
         documento: graficaría cuánto se escribió, no sobre qué.
      3. TruncatedSVD sobre TF-IDF ES Latent Semantic Analysis (Deerwester et
         al., 1990). Las componentes latentes agrupan términos co-ocurrentes, de
         modo que dos sinónimos dejan de ser ortogonales. Es el único punto del
         pipeline que merece el adjetivo "semántico".
    """
    n_componentes = min(n_componentes, X.shape[1] - 1)   # SVD truncada exige k < n_features
    svd = TruncatedSVD(n_components=n_componentes, random_state=semilla)
    coords = svd.fit_transform(X)                        # opera sobre sparse, sin densificar
    return coords, svd.explained_variance_ratio_


def proyectar_pca(M, n_componentes: int = 2, semilla: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Proyecta una matriz DENSA (embeddings) con PCA.

    Es el caso opuesto al de proyectar(). Allá la matriz es dispersa y
    centrarla la densificaría; aquí ya es densa y centrarla no cuesta nada.
    Además centrar es necesario: los embeddings de Word2Vec comparten un
    vector medio no nulo, y sin restarlo (TruncatedSVD) la primera componente
    apunta hacia esa media, que es común a todo el vocabulario y no distingue
    ninguna palabra de otra.

    La varianza explicada es relativa a las filas de M, es decir, a las
    palabras que se grafican, no al vocabulario completo.
    """
    pca = PCA(n_components=n_componentes, random_state=semilla)
    coords = pca.fit_transform(np.asarray(M))
    return coords, pca.explained_variance_ratio_


def graficar(coords, etiquetas, varianza, titulo: str, ruta: Path,
             anotaciones: Sequence[str] | None = None,
             titulo_leyenda: str = "Corpus de origen",
             prefijo_eje: str = "Componente latente") -> Path:
    """
    Scatter 2D con la varianza explicada EN EL TÍTULO.

    Reportar explained_variance_ratio_ no es burocracia estadística: si el
    gráfico representa el 12% de la información, cualquier agrupamiento visible
    puede ser artefacto de proyección. El número es la defensa contra el propio
    sistema visual del lector.

    `anotaciones` rotula cada punto (las palabras, en la etapa 3). Los demás
    parámetros opcionales conservan por defecto el gráfico de la etapa 2.
    """
    ruta = Path(ruta)
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

    for clase in sorted(set(etiquetas)):
        m = np.array([e == clase for e in etiquetas])    # máscara booleana por clase
        ax.scatter(coords[m, 0], coords[m, 1],
                   s=80, alpha=0.85, edgecolors="k", linewidths=0.5, label=clase)

    if anotaciones is not None:
        for (x, y), texto in zip(coords, anotaciones):
            ax.annotate(texto, (x, y), xytext=(4, 4), textcoords="offset points",
                        fontsize=8)

    ax.axhline(0, color="grey", lw=0.5, ls="--")         # ejes de referencia en el origen
    ax.axvline(0, color="grey", lw=0.5, ls="--")
    ax.set_xlabel(f"{prefijo_eje} 1 ({varianza[0]:.1%} de la varianza)")
    ax.set_ylabel(f"{prefijo_eje} 2 ({varianza[1]:.1%} de la varianza)")
    ax.set_title(f"{titulo}\nVarianza explicada acumulada: {varianza.sum():.1%}")
    ax.legend(title=titulo_leyenda)

    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)      # sin esto, N figuras abiertas son una fuga de memoria en lotes
    return ruta
