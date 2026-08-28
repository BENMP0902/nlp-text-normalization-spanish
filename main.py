"""Orquestador del módulo 3. Punto de entrada único.

Uso:
    python main.py
    python main.py --sin-negadores     # reproduce el pipeline del curso
    python main.py --sin-filtro        # sin filtrado de stop words
"""
import argparse

from src import config
from src.carga import cargar_texto, diagnostico
from src.normalizacion import (
    procesar,
    lematizar,
    construir_corpus,
    tokens_de_contenido,
    limpiar_lema,
)
from src.metricas import compresion_vocabulario, tabla_stem_vs_lema

# Palabras elegidas para la comparativa cualitativa: verbos conjugados y
# sustantivos derivados, donde stemming y lematización divergen de verdad.
PALABRAS_MUESTRA = {
    "tenía", "representaba", "tragaba", "personas", "mayores",
    "explicaciones", "dibujos", "comprender", "aconsejaron",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalización y lematización (Módulo 3)"
    )
    # action="store_true": la bandera no lleva valor; su presencia la activa.
    parser.add_argument("--sin-negadores", action="store_true",
                        help="Filtra también los negadores (comportamiento del curso)")
    parser.add_argument("--sin-filtro", action="store_true",
                        help="Desactiva por completo el filtrado de stop words")
    args = parser.parse_args()

    # Sobrescribimos config desde la CLI: permite comparar variantes sin editar
    # código. argparse convierte --sin-negadores en args.sin_negadores (guion
    # bajo), porque los guiones no son válidos en identificadores de Python.
    if args.sin_negadores:
        config.PRESERVAR_NEGADORES = False
    if args.sin_filtro:
        config.FILTRAR_STOPWORDS = False

    # --- 1. Carga ---
    texto = cargar_texto(config.ARCHIVO_CORPUS)
    print("Diagnóstico de carga:", diagnostico(texto))

    # --- 2. Pipeline ---
    doc = procesar(texto)
    print(f"\nTokens totales (incluye puntuación y espacios): {len(doc)}")
    print("Primeros 20 tokens:", [t.text for t in doc][:20])

    tokens = tokens_de_contenido(doc)
    formas_contenido = [t.text for t in tokens]
    lemas = [limpiar_lema(t.lemma_.lower()) for t in tokens]
    _, transformaciones = lematizar(doc)

    # Base honesta: excluimos puntuación y espacios del denominador. Contar los
    # 39 saltos de línea del archivo como "tokens eliminados" infla la cifra.
    utiles = [t for t in doc if not (t.is_punct or t.is_space)]
    print(f"\nTokens totales (con puntuación y espacios): {len(doc)}")
    print(f"Tokens útiles (sin puntuación ni espacios):  {len(utiles)}")
    print(f"Tokens de contenido (tras filtrar stop words): {len(tokens)} "
          f"— filtrado elimina {1 - len(tokens)/len(utiles):.1%} del texto útil")
    print("Transformaciones (forma, POS, lema):", transformaciones[:10])

    # --- 4. Métricas: ahora sobre la MISMA base ---
    print("\n=== Compresión de vocabulario (solo tokens de contenido) ===")
    print(compresion_vocabulario(formas_contenido, lemas).to_string(index=False))

    # --- 5. Corpus por oración (insumo del módulo 4) ---
    corpus = construir_corpus(doc)
    print(f"\nOraciones con contenido: {len(corpus)}")
    print("Primera oración lematizada:", corpus[0] if corpus else "—")


if __name__ == "__main__":     # solo ejecuta al invocar, no al importar
    main()
