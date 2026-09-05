"""
vectorizacion.py — Feature Extraction: de tokens a espacio vectorial.

Implementa las dos representaciones del módulo 4 (Bag-of-Words y TF-IDF)
respetando la tokenización producida en el módulo 3.
"""
from __future__ import annotations
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _identidad(documento: List[str]) -> List[str]:
    """
    Analyzer de identidad: recibe una lista de tokens y la devuelve intacta.

    Se define a nivel de módulo (no como lambda) porque un vectorizador con
    lambda NO es serializable con pickle/joblib. Si mañana quieres persistir el
    vectorizador entrenado para un servicio de inferencia, la lambda te bloquea.
    Costo de hacerlo bien hoy: cero. Costo de descubrirlo en producción: una tarde.
    """
    return documento


def construir_bow(corpus_tokens: Sequence[List[str]], min_df: int = 2) -> Tuple:
    """
    Bag-of-Words: matriz de CONTEOS crudos, shape (n_documentos, |V|).

    Sin normalización: el valor de la celda es literalmente cuántas veces
    aparece el término en el documento. Es el baseline contra el que vamos a
    comparar TF-IDF.
    """
    vectorizador = CountVectorizer(
        analyzer=_identidad,   # NO re-tokeniza: consume la lista de lemas tal cual
        lowercase=False,       # ya normalizaste en el módulo 3; doble trabajo = doble riesgo
        min_df=min_df,         # descarta hápax legómena (ley de Zipf: son ~50% del vocab)
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
    documentos obtiene idf = 1.0, NO 0.0. El IDF de sklearn **no anula** nada.
    Por eso `max_df` es necesario: ese sí elimina los términos ubicuos.
    Ref.: scikit-learn, User Guide → "Tf–idf term weighting".
    """
    vectorizador = TfidfVectorizer(
        analyzer=_identidad,
        lowercase=False,
        min_df=min_df,
        max_df=max_df,        # descarta términos en >90% de los documentos
        sublinear_tf=True,    # tf → 1 + log(tf): amortigua Zipf. Casi siempre mejora.
        norm="l2",            # default explícito: filas en la esfera unitaria
        smooth_idf=True,      # default explícito: documenta la fórmula que usas
    )
    X = vectorizador.fit_transform(corpus_tokens)
    return X, vectorizador


def diagnosticar_idf(vectorizador) -> dict:
    """
    Comprobación de sanidad (sanity check) del IDF.

    Si std ≈ 0, todos los términos pesan igual → TF-IDF degeneró a BoW binario
    normalizado → tus documentos son demasiado cortos o demasiado pocos.
    Este diagnóstico es la diferencia entre "corrí el código" y "sé qué salió".
    """
    idf = vectorizador.idf_
    return {
        "vocabulario": len(vectorizador.get_feature_names_out()),
        "idf_min": float(idf.min()),
        "idf_max": float(idf.max()),
        "idf_std": float(idf.std()),
    }


def terminos_discriminantes(X, vectorizador, etiquetas: Sequence[str], top_n: int = 15):
    """
    Para cada clase (libro), los términos con mayor peso TF-IDF promedio.

    Es la lectura INTERPRETABLE del modelo: puedes señalar exactamente qué
    término empujó la decisión. Un embedding denso no te da esto — y es la
    razón por la que TF-IDF sigue vivo donde hay que auditar decisiones.
    """
    vocab = np.array(vectorizador.get_feature_names_out())
    etiquetas = np.array(etiquetas)
    filas = []
    for clase in np.unique(etiquetas):
        mascara = etiquetas == clase                  # bool array: qué filas son de esta clase
        media = np.asarray(X[mascara].mean(axis=0)).ravel()   # peso medio por término
        top = media.argsort()[::-1][:top_n]           # argsort asc → invierte → toma top_n
        for rango, idx in enumerate(top, start=1):
            filas.append({"clase": clase, "rango": rango,
                          "termino": vocab[idx], "peso_medio": round(float(media[idx]), 4)})
    return pd.DataFrame(filas)


def matriz_similitud(X) -> np.ndarray:
    """
    Similitud coseno entre documentos.

    Coseno y NO distancia euclidiana: mide ÁNGULO, ignora magnitud, así que un
    párrafo largo y uno corto sobre el mismo tema quedan cerca.
    (Con TF-IDF L2-normalizado ambas ordenan igual; con BoW crudo NO.)
    Ref.: Salton, Wong & Yang (1975), "A Vector Space Model for Automatic Indexing".
    """
    return cosine_similarity(X)


def par_mas_similar(S: np.ndarray) -> Tuple[int, int, float]:
    """Par de documentos distintos con mayor similitud."""
    S = S.copy()
    np.fill_diagonal(S, -1.0)          # anula la autosimilitud (siempre 1.0)
    i, j = np.unravel_index(S.argmax(), S.shape)   # índice plano → (fila, columna)
    return int(i), int(j), float(S[i, j])
