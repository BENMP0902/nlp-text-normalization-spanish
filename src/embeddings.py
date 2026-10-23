"""
embeddings.py — Etapa 3 (módulo 5): semántica distribucional con Word2Vec.

Distinción frente a vectorizacion.py: allá cada DOCUMENTO es un vector disperso
de conteos; aquí cada PALABRA es un vector denso aprendido de sus contextos
(hipótesis distribucional: Harris, 1954; Firth, 1957).

No reimplementa la etapa 1: segmenta con segmentacion.py y lematiza con
construir_corpus() de normalizacion.py, con una sola diferencia deliberada,
que es conservar las stop words.
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from gensim.models import Word2Vec

from src import config
from src.normalizacion import (
    cargar_modelo,
    construir_corpus,
    es_contenido,
    limpiar_lema,
    procesar_por_partes,
)
from src.segmentacion import segmentar_en_parrafos

# Categorías que cuentan como palabra de contenido al GRAFICAR. El
# entrenamiento no filtra nada; esto solo decide qué se muestra.
_POS_CONTENIDO = frozenset({"NOUN", "PROPN", "VERB", "ADJ"})

# Conjunto vacío: es_contenido descarta puntuación, espacios y números, y deja
# pasar todas las palabras. Es la ventana de contexto íntegra que Word2Vec
# necesita; las palabras frecuentes las degrada el submuestreo (`sample`).
_SIN_FILTRO = frozenset()


def preparar_corpus(texto: str) -> Tuple[List[List[str]], Dict[str, Counter]]:
    """Texto crudo → oraciones lematizadas CON stop words, más el POS de cada lema.

    Trocea en párrafos (líneas en blanco) y nunca a mitad de oración.
    min_caracteres=1 conserva todos los párrafos: a diferencia del IDF, a
    Word2Vec una línea de diálogo corta le aporta contexto, no ruido.

    El conteo de POS por lema se usa solo para colorear la figura. Se toma en
    la misma pasada porque el Doc no se conserva: procesar_por_partes es un
    generador.
    """
    parrafos = segmentar_en_parrafos(texto, min_caracteres=1)
    oraciones: List[List[str]] = []
    pos_por_lema: Dict[str, Counter] = defaultdict(Counter)

    for doc in procesar_por_partes(parrafos):
        oraciones.extend(construir_corpus(doc, stop_words=_SIN_FILTRO))
        for t in doc:
            if es_contenido(t, _SIN_FILTRO):
                pos_por_lema[limpiar_lema(t.lemma_.lower())][t.pos_] += 1

    return oraciones, pos_por_lema


def verificar_semilla_hash() -> str | None:
    """Advierte si PYTHONHASHSEED no está fijado.

    gensim inicializa cada vector a partir de hash(palabra + semilla), y el
    hash de str en Python se aleatoriza por proceso. Fijar `seed` sin fijar
    PYTHONHASHSEED da vectores distintos en cada corrida. La variable se lee
    al arrancar el intérprete: asignarla en os.environ desde aquí no surte
    efecto, así que solo se puede avisar.
    """
    if os.environ.get("PYTHONHASHSEED") is None:
        return ("[aviso] PYTHONHASHSEED no está definido: los vectores cambiarán "
                "entre corridas. Ejecuta: PYTHONHASHSEED=0 python main.py --embeddings")
    return None


def entrenar(oraciones: Sequence[List[str]], semilla: int = config.W2V_SEMILLA) -> Word2Vec:
    """Entrena Word2Vec con los hiperparámetros de config.

    workers=1 es obligatorio para la reproducibilidad: con varios hilos el
    orden en que se aplican las actualizaciones depende del planificador del
    sistema operativo, y la semilla no lo controla.
    """
    return Word2Vec(
        sentences=oraciones,
        vector_size=config.W2V_VECTOR_SIZE,
        window=config.W2V_VENTANA,
        min_count=config.W2V_MIN_COUNT,
        sample=config.W2V_SAMPLE,
        sg=config.W2V_SG,
        epochs=config.W2V_EPOCHS,
        seed=semilla,
        workers=config.W2V_WORKERS,
    )


def stop_words_para_graficar() -> frozenset:
    """Lista COMPLETA de spaCy, negadores incluidos.

    En las etapas 1 y 2 los negadores se preservan porque invierten la
    polaridad de un documento. En una figura de semántica léxica, "no" y
    "nunca" son palabras función como cualquier otra y taparían el contenido.
    """
    nlp = cargar_modelo(config.MODELO_SPACY, config.COMPONENTES_DESACTIVADOS)
    return frozenset(nlp.Defaults.stop_words)


def palabras_de_contenido(modelo: Word2Vec, pos_por_lema: Dict[str, Counter],
                          n: int | None = None) -> List[str]:
    """Vocabulario del modelo sin stop words, por frecuencia descendente.

    gensim ya ordena index_to_key por frecuencia, así que basta filtrar. Se
    exige POS dominante de contenido: el filtro por lista no atrapa adverbios
    y formas residuales que el lematizador deja fuera de la lista. isalpha()
    descarta abreviaturas ('d.', 'sr.') que el tagger marca como PROPN.
    """
    stop = stop_words_para_graficar()
    palabras = [
        w for w in modelo.wv.index_to_key
        if w not in stop and len(w) > 1 and w.isalpha()
        and pos_dominante(w, pos_por_lema) in _POS_CONTENIDO
    ]
    return palabras if n is None else palabras[:n]


def pos_dominante(lema: str, pos_por_lema: Dict[str, Counter]) -> str:
    """POS más frecuente del lema en el corpus ('X' si no se observó)."""
    conteo = pos_por_lema.get(lema)
    return conteo.most_common(1)[0][0] if conteo else "X"


def vecinos(modelo: Word2Vec, palabra: str, k: int = config.W2V_TOP_K) -> List[str]:
    """Top-k vecinos por coseno, sin las similitudes."""
    return [w for w, _ in modelo.wv.most_similar(palabra, topn=k)]


def diagnosticar_suficiencia(modelo: Word2Vec, oraciones: Sequence[List[str]],
                             sondas: Sequence[str]) -> dict:
    """Mide si el corpus alcanza para que los vecinos signifiquen algo.

    most_similar SIEMPRE devuelve una lista ordenada con cuatro decimales,
    incluso cuando los vectores casi no se han movido de su inicialización
    aleatoria. Nada en la salida distingue la señal del ruido, así que aquí se
    mide directamente: se reentrena con OTRA semilla y se compara el top-k de
    cada sonda (índice de Jaccard). Lo que pertenece al corpus sobrevive al
    cambio de semilla; lo que pertenece a la inicialización, no.
    """
    total = sum(len(o) for o in oraciones)
    en_vocabulario = sum(1 for o in oraciones for w in o if w in modelo.wv.key_to_index)

    alterno = entrenar(oraciones, semilla=config.W2V_SEMILLA + 1)
    jaccard = {}
    for s in sondas:
        a, b = set(vecinos(modelo, s)), set(vecinos(alterno, s))
        jaccard[s] = len(a & b) / len(a | b)
    estabilidad = float(np.mean(list(jaccard.values()))) if jaccard else 0.0

    avisos = []
    if total < config.W2V_MIN_TOKENS:
        avisos.append(
            f"[aviso] Corpus insuficiente: {total:,} tokens (umbral "
            f"{config.W2V_MIN_TOKENS:,}). Los vectores de palabras poco "
            f"frecuentes están dominados por su inicialización."
        )
    if estabilidad < config.W2V_MIN_ESTABILIDAD:
        avisos.append(
            f"[aviso] Vecinos inestables: Jaccard medio {estabilidad:.2f} entre "
            f"semillas {config.W2V_SEMILLA} y {config.W2V_SEMILLA + 1} (umbral "
            f"{config.W2V_MIN_ESTABILIDAD}). most_similar está reportando ruido "
            f"de la inicialización, no relaciones del corpus."
        )

    return {
        "tokens": total,
        "oraciones": len(oraciones),
        "vocabulario": len(modelo.wv),
        "cobertura": round(en_vocabulario / total, 3) if total else 0.0,
        "estabilidad_jaccard": round(estabilidad, 3),
        "jaccard_por_sonda": {k: round(v, 2) for k, v in jaccard.items()},
        "avisos": avisos,
    }


def vecindarios(modelo: Word2Vec, sondas: Sequence[str], permitidas: set[str],
                k: int = config.W2V_TOP_K) -> Tuple[List[str], List[str]]:
    """Cada sonda con sus k vecinos de contenido, etiquetados por sonda.

    Se piden más vecinos de los necesarios y se filtran a `permitidas`
    (contenido) para que la figura no se llene de palabras función. Una
    palabra vecina de dos sondas se asigna a la primera: un punto no puede
    tener dos colores.
    """
    palabras, etiquetas, vistas = [], [], set()
    for s in sondas:
        candidatas = [s] + [w for w in vecinos(modelo, s, k=10 * k) if w in permitidas]
        tomadas = [w for w in candidatas if w not in vistas][:k + 1]
        palabras.extend(tomadas)
        etiquetas.extend([s] * len(tomadas))
        vistas.update(tomadas)
    return palabras, etiquetas


def matriz_de(modelo: Word2Vec, palabras: Sequence[str]) -> np.ndarray:
    """Apila los vectores de las palabras, en el mismo orden."""
    return np.vstack([modelo.wv[w] for w in palabras])


def guardar_texto_procesado(oraciones: Sequence[List[str]], ruta: Path) -> Path:
    """Una oración por línea y lemas separados por espacio.

    Es el formato LineSentence de gensim y la entrada EXACTA del
    entrenamiento: Word2Vec(sentences=LineSentence(ruta)) reproduce el modelo
    sin pasar otra vez por spaCy. (corpus_file= no: usa otra implementación
    multihilo y no garantiza los mismos vectores.)
    """
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(" ".join(o) for o in oraciones) + "\n", encoding="utf-8")
    return ruta
