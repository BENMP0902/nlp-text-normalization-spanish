"""Medición del efecto de la normalización.

Existe porque la comparación del curso (`stem == lema`) no mide nada: stem y
lema viven en espacios de salida distintos —el stem es un truncamiento, el lema
es una palabra real— así que la no-coincidencia es LO ESPERADO, no información.
Lo que sí importa: cuánto se comprime el vocabulario, a qué costo, y si el
corpus tiene siquiera redundancia que comprimir.
"""
from collections import defaultdict

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

    v_original = {f.lower() for f in formas}      # set comprehension: únicos
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


def type_token_ratio(formas: list[str]) -> float:
    """TTR: tipos únicos / tokens totales.

    Mide la redundancia léxica del corpus. Cercano a 1.0 = casi todas las
    palabras aparecen una sola vez (hapax legomena), y entonces NINGUNA técnica
    de normalización puede comprimir mucho: no hay variantes que fusionar.
    Es el número que contextualiza cualquier métrica de compresión.

    Advertencia: la TTR depende del tamaño del texto — decrece al crecer el
    corpus. No compares TTR entre textos de longitud distinta sin normalizar
    (existen variantes como la TTR estandarizada, STTR, justamente por esto).
    """
    if not formas:
        raise ValueError("Lista de formas vacía: TTR indefinida.")
    return len({f.lower() for f in formas}) / len(formas)


def sobre_stemming(formas: list[str], lemas: list[str]) -> pd.DataFrame:
    """Casos donde el stemmer fusionó formas que el lematizador separó.

    Esta es la EVIDENCIA del over-stemming, no su afirmación. Agrupamos por
    stem; si un grupo contiene más de un lema distinto, el stemmer colapsó
    conceptos que el análisis morfológico distingue.

    Ejemplo típico en español: 'dibujo' (NOUN) y 'dibujar' (VERB) comparten el
    stem 'dibuj'. Fusionarlos destruye la distinción entre el objeto y la
    acción — la misma que el capítulo entero está discutiendo.
    """
    if len(formas) != len(lemas):
        raise ValueError("formas y lemas deben venir del mismo conjunto.")

    # defaultdict(set): al acceder a una clave inexistente crea un set vacío,
    # así evitamos el patrón `if k not in d: d[k] = set()` en cada iteración.
    por_stem_formas = defaultdict(set)
    por_stem_lemas = defaultdict(set)

    for forma, lema in zip(formas, lemas):        # zip los recorre en paralelo
        stem = _stemmer.stem(forma.lower())
        por_stem_formas[stem].add(forma.lower())
        por_stem_lemas[stem].add(lema)

    filas = [
        {
            "stem": stem,
            "formas_fusionadas": ", ".join(sorted(por_stem_formas[stem])),
            "lemas_distintos": ", ".join(sorted(lemas_set)),
        }
        for stem, lemas_set in por_stem_lemas.items()
        if len(lemas_set) > 1                     # solo los grupos conflictivos
    ]
    return pd.DataFrame(filas)
