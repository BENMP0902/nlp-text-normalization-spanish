"""Medición del efecto de la normalización.

Existe porque la comparación del curso (`stem == lema`) no mide nada: stem y
lema viven en espacios de salida distintos —el stem es un truncamiento, el lema
es una palabra real— así que la no-coincidencia es LO ESPERADO, no información.
Lo que sí importa: cuánto se comprime el vocabulario y a qué costo.
"""
import pandas as pd
from nltk.stem import SnowballStemmer

# Snowball (Porter2) para español. Es algorítmico: no requiere descargar
# recursos de NLTK, a diferencia de los tokenizadores (punkt) o wordnet.
_stemmer = SnowballStemmer("spanish")


def compresion_vocabulario(formas: list[str], lemas: list[str]) -> pd.DataFrame:
    """Compara |V| bajo tres tratamientos SOBRE EL MISMO conjunto de tokens.

    CONTRATO: `formas` y `lemas` deben tener la misma longitud y provenir de
    los mismos tokens, en el mismo orden. Si `formas` incluye stop words y
    `lemas` no, la comparación es inválida: la columna de lematización estaría
    midiendo filtrado, no normalización.

    |V| (cardinalidad del vocabulario) es la métrica directa de dimensionalidad:
    cada término único será una columna de tu matriz término-documento.
    Menos |V| = menos sparsity. Pero MÁS compresión no es mejor por sí sola:
    el stemmer comprime más porque sobre-agrupa (over-stemming), fusionando
    conceptos distintos que comparten prefijo.
    """
    if len(formas) != len(lemas):
        raise ValueError(
            f"Bases distintas: {len(formas)} formas vs {len(lemas)} lemas. "
            "Ambas listas deben venir del mismo conjunto de tokens."
        )
    if not formas:
        raise ValueError("Lista de formas vacía: nada que medir.")

    v_original = {f.lower() for f in formas}
    v_stem = {_stemmer.stem(f) for f in formas}
    v_lema = set(lemas)

    base = len(v_original)
    return pd.DataFrame([
        {"tratamiento": "sin normalizar", "|V|": base, "reduccion": 0.0},
        {"tratamiento": "stemming", "|V|": len(v_stem),
         "reduccion": 1 - len(v_stem) / base},
        {"tratamiento": "lematización", "|V|": len(v_lema),
         "reduccion": 1 - len(v_lema) / base},
    ])

def tabla_stem_vs_lema(doc, palabras: set[str]) -> pd.DataFrame:
    """Comparativa cualitativa sobre palabras seleccionadas.

    La columna que importa NO es si stem y lema coinciden. Es si el lema es una
    palabra real del español y el stem no. Ahí se ve el trade-off: el stemmer
    es rápido y agresivo; el lematizador es lento y correcto.
    """
    filas = []
    vistos = set()
    for token in doc:
        forma = token.text.lower()
        if forma in palabras and forma not in vistos:
            vistos.add(forma)                     # una fila por palabra única
            filas.append({
                "original": token.text,
                "POS": token.pos_,                # Universal POS tag
                "stem": _stemmer.stem(token.text),
                "lema": token.lemma_.lower(),
            })
    return pd.DataFrame(filas)
