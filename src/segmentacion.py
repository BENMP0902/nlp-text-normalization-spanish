"""
segmentacion.py — Convierte texto crudo en una lista de DOCUMENTOS.

Responsabilidad única (Single Responsibility): decidir qué unidad de texto
constituye un documento. Es la decisión que gobierna el comportamiento del IDF,
así que vive en su propio módulo y no escondida dentro del vectorizador.
"""
from __future__ import annotations          # anotaciones perezosas (PEP 563); útil <3.10
import re
from typing import List

# Un párrafo se separa por una línea en blanco: \n, espacios opcionales, \n.
# Compilamos el patrón una sola vez a nivel de módulo (se reutiliza en cada llamada).
_SEP_PARRAFO = re.compile(r"\n\s*\n")

# Colapsa cualquier racha de whitespace (saltos de línea internos, tabs) a un espacio.
_WHITESPACE = re.compile(r"\s+")


def segmentar_en_parrafos(texto: str, min_caracteres: int = 150) -> List[str]:
    """
    Parte el texto por líneas en blanco.

    min_caracteres descarta fragmentos demasiado cortos (títulos, números de
    capítulo, líneas sueltas de diálogo). Documentos muy cortos reintroducen
    exactamente el problema que queremos evitar: df≈1 e IDF sin varianza.
    """
    bloques = _SEP_PARRAFO.split(texto)                       # lista de bloques crudos
    limpios = [_WHITESPACE.sub(" ", b).strip() for b in bloques]  # normaliza whitespace
    return [p for p in limpios if len(p) >= min_caracteres]   # filtra por longitud


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


def construir_documentos(texto: str, nlp, min_documentos: int = 8) -> List[str]:
    """
    Estrategia con degradación controlada (graceful degradation):
    1) intenta párrafos;
    2) si salen pocos, cae a bloques de oraciones.

    Devolver pocos documentos no lanza excepción, pero arruina el IDF en
    silencio. Por eso la verificación es explícita y no un supuesto.
    """
    documentos = segmentar_en_parrafos(texto)
    if len(documentos) >= min_documentos:
        return documentos

    # El .txt no tenía estructura de párrafos utilizable → plan B.
    print(f"[segmentacion] Solo {len(documentos)} párrafos; uso bloques de oraciones.")
    return segmentar_en_bloques(nlp(texto))
