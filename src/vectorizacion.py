"""
vectorizacion.py — Feature Extraction: de tokens lematizados a espacio vectorial.

Implementa las dos representaciones del módulo 4 (Bag-of-Words y TF-IDF)
respetando la tokenización producida en la etapa 1. No reimplementa nada de
normalizacion.py: consume su salida.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _identidad(documento: List[str]) -> List[str]:
    """
    Analyzer de identidad: recibe una lista de tokens y la devuelve intacta.

    Por qué existe: CountVectorizer y TfidfVectorizer aplican por defecto
    token_pattern=r"(?u)\\b\\w\\w+\\b" y lowercase=True sobre la cadena que
    reciben. Pasarles " ".join(lemas) descarta todo token de un solo carácter y
    parte los lemas con guion, anulando en silencio el criterio lingüístico de
    la etapa 1. Con este analyzer, sklearn no toca la tokenización.

    Por qué es función nombrada y no lambda: un vectorizador con lambda NO es
    serializable con pickle/joblib. Persistir el vectorizador entrenado para un
    servicio de inferencia es un requisito frecuente; el costo de preverlo hoy
    es cero.
    """
    return documento


def construir_bow(corpus_tokens: Sequence[List[str]], min_df: int = 2) -> Tuple:
    """
    Bag-of-Words: matriz de CONTEOS crudos, shape (n_documentos, |V|).

    Sin normalización: el valor de cada celda es literalmente cuántas veces
    aparece el término en el documento. Es el baseline contra el que se compara
    TF-IDF.
    """
    vectorizador = CountVectorizer(
        analyzer=_identidad,   # no re-tokeniza: consume la lista de lemas tal cual
        lowercase=False,       # la normalización ya ocurrió en la etapa 1
        min_df=min_df,         # descarta hapax legomena (ley de Zipf: ~50% del vocabulario)
    )
    X = vectorizador.fit_transform(corpus_tokens)   # scipy.sparse.csr_matrix
    return X, vectorizador


def construir_tfidf(
    corpus_tokens: Sequence[List[str]],
    min_df: int = 2,
    max_df: float = 0.90,
) -> Tuple:
    """
    TF-IDF con parámetros elegidos, no heredados del default.

    OJO con la fórmula real de scikit-learn (smooth_idf=True, el default):

        idf(t) = ln[ (1 + n) / (1 + df(t)) ] + 1

    El "+1" fuera del logaritmo implica que un término presente en TODOS los
    documentos obtiene idf = 1.0, NO 0.0. El IDF de sklearn no anula nada; por
    eso max_df es necesario.
    Ref.: scikit-learn, User Guide → "Tf-idf term weighting".
    """
    vectorizador = TfidfVectorizer(
        analyzer=_identidad,
        lowercase=False,
        min_df=min_df,
        max_df=max_df,        # descarta términos presentes en >90% de los documentos
        sublinear_tf=True,    # tf → 1 + log(tf): amortigua la ley de Zipf
        norm="l2",            # default explícito: filas sobre la esfera unitaria
        smooth_idf=True,      # default explícito: documenta qué fórmula se usa
    )
    X = vectorizador.fit_transform(corpus_tokens)
    return X, vectorizador


def diagnosticar(X, vectorizador) -> dict:
    """
    Comprobación de sanidad (sanity check) del espacio construido.

    Si idf_std ≈ 0, todos los términos pesan igual → TF-IDF degeneró a BoW
    binario normalizado → los documentos son demasiado cortos o demasiado pocos.
    Este diagnóstico es la diferencia entre "corrí el código" y "sé qué salió".
    """
    idf = getattr(vectorizador, "idf_", None)
    reporte = {
        "documentos": X.shape[0],
        "vocabulario": X.shape[1],
        "densidad_pct": round(100 * X.nnz / (X.shape[0] * X.shape[1]), 3),
    }
    if idf is not None:
        reporte |= {
            "idf_min": round(float(idf.min()), 4),
            "idf_max": round(float(idf.max()), 4),
            "idf_std": round(float(idf.std()), 4),
        }
    return reporte


def terminos_discriminantes(X, vectorizador, etiquetas: Sequence[str], top_n: int = 15):
    """
    Para cada clase (corpus), los términos con mayor peso TF-IDF promedio.

    Es la lectura INTERPRETABLE del modelo: se puede señalar exactamente qué
    término empujó cada decisión. Un embedding denso no permite esto, y es una
    de las razones por las que las representaciones dispersas siguen vigentes.
    """
    vocab = np.array(vectorizador.get_feature_names_out())
    etiquetas = np.array(etiquetas)
    filas = []
    for clase in np.unique(etiquetas):
        mascara = etiquetas == clase                            # máscara booleana de filas
        media = np.asarray(X[mascara].mean(axis=0)).ravel()     # peso medio por término
        top = media.argsort()[::-1][:top_n]                     # argsort asc → invertir → top_n
        for rango, idx in enumerate(top, start=1):
            filas.append({
                "clase": clase,
                "rango": rango,
                "termino": vocab[idx],
                "peso_medio": round(float(media[idx]), 4),
            })
    return pd.DataFrame(filas)


def matriz_similitud(X) -> np.ndarray:
    """
    Similitud coseno entre documentos.

    Coseno y NO distancia euclidiana: mide ÁNGULO e ignora magnitud, así que un
    párrafo largo y uno corto sobre el mismo tema quedan cerca. Con TF-IDF
    L2-normalizado ambas métricas ordenan igual; con BoW crudo NO.
    Ref.: Salton, Wong & Yang (1975), "A Vector Space Model for Automatic Indexing".
    """
    return cosine_similarity(X)


def par_mas_similar(S: np.ndarray) -> Tuple[int, int, float]:
    """Par de documentos DISTINTOS con mayor similitud."""
    S = S.copy()
    np.fill_diagonal(S, -1.0)                       # anula la autosimilitud (siempre 1.0)
    i, j = np.unravel_index(S.argmax(), S.shape)    # índice plano → (fila, columna)
    return int(i), int(j), float(S[i, j])


def similitud_intra_vs_inter(S: np.ndarray, etiquetas: Sequence[str]) -> dict:
    """
    Contrasta la similitud media entre documentos del mismo corpus contra la de
    documentos de corpus distintos.

    Es la prueba cuantitativa de la hipótesis del experimento. Si intra no supera
    claramente a inter, la vectorización no capturó contenido temático y el
    gráfico no debe interpretarse.
    """
    etiquetas = np.array(etiquetas)
    mismo = etiquetas[:, None] == etiquetas[None, :]     # matriz booleana (n, n)
    fuera_diagonal = ~np.eye(len(etiquetas), dtype=bool)  # excluye la autosimilitud
    return {
        "intra_corpus": round(float(S[mismo & fuera_diagonal].mean()), 4),
        "inter_corpus": round(float(S[~mismo].mean()), 4),
    }
