"""
segmentacion.py — Convierte texto crudo en una lista de DOCUMENTOS.

Responsabilidad única: decidir qué unidad de texto constituye un documento.
Es la decisión que gobierna el comportamiento del IDF, así que vive en su propio
módulo y no escondida dentro del vectorizador.

Distinción frente a normalizacion.py: ahí se segmenta en ORACIONES, que es una
operación lingüística dependiente del parser de spaCy. Aquí se segmenta en
DOCUMENTOS, que es una decisión de diseño experimental.
"""
from __future__ import annotations

import re
from typing import List

# Un párrafo se delimita por una línea en blanco: \n, whitespace opcional, \n.
# Se compila una sola vez a nivel de módulo: el patrón se reutiliza en cada llamada.
_SEP_PARRAFO = re.compile(r"\n\s*\n")

# Colapsa cualquier racha de whitespace (saltos internos, tabs) a un solo espacio.
_WHITESPACE = re.compile(r"\s+")


def segmentar_en_parrafos(texto: str, min_caracteres: int = 150) -> List[str]:
    """
    Parte el texto por líneas en blanco.

    `min_caracteres` descarta fragmentos demasiado cortos (títulos, números de
    capítulo, líneas sueltas de diálogo). Documentos muy cortos reintroducen
    exactamente el problema que queremos evitar: df≈1 e IDF sin varianza.
    """
    bloques = _SEP_PARRAFO.split(texto)                            # bloques crudos
    limpios = [_WHITESPACE.sub(" ", b).strip() for b in bloques]   # whitespace normalizado
    return [p for p in limpios if len(p) >= min_caracteres]        # filtro por longitud


def segmentar_en_bloques(doc, oraciones_por_bloque: int = 4) -> List[str]:
    """
    FALLBACK: agrupa oraciones de a N usando el sentencizer de spaCy.

    Se usa cuando el .txt viene sin líneas en blanco (texto "wrapped" a 80
    columnas), caso en el que segmentar_en_parrafos devolvería 1 solo documento.

    `doc` es un spacy.tokens.Doc ya procesado.
    """
    oraciones = [s.text.strip() for s in doc.sents if s.text.strip()]
    # Slicing con paso: [0:4], [4:8], [8:12]... Idioma estándar de chunking en Python.
    return [
        " ".join(oraciones[i:i + oraciones_por_bloque])
        for i in range(0, len(oraciones), oraciones_por_bloque)
    ]


def construir_documentos(texto: str, nlp, min_documentos: int = 6,
                         oraciones_por_bloque: int = 4) -> List[str]:
    """
    Estrategia con degradación controlada (graceful degradation):
      1) intenta párrafos;
      2) si salen pocos, cae a bloques de oraciones.

    Devolver pocos documentos no lanza ninguna excepción, pero arruina el IDF en
    silencio. Por eso la verificación es explícita y se reporta por consola.
    """
    documentos = segmentar_en_parrafos(texto)
    if len(documentos) >= min_documentos:
        print(f"[segmentacion] {len(documentos)} documentos por párrafo.")
        return documentos

    print(
        f"[segmentacion] Solo {len(documentos)} párrafos "
        f"(mínimo {min_documentos}); uso bloques de oraciones."
    )
    documentos = segmentar_en_bloques(nlp(texto), oraciones_por_bloque)
    print(f"[segmentacion] {len(documentos)} documentos por bloque de oraciones.")
    return documentos
