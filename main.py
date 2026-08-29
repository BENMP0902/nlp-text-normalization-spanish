import argparse
from collections import Counter
from pathlib import Path

from src import config
from src.carga import cargar_texto, diagnostico
from src.normalizacion import (
    procesar,
    lematizar,
    construir_corpus,
    tokens_de_contenido,
    limpiar_lema,
)
from src.metricas import (
    compresion_vocabulario,
    tabla_stem_vs_lema,
    type_token_ratio,
    sobre_stemming,
)


def resolver_corpus(indicado: Path | None) -> Path:
    """Convierte lo que el usuario escribió en una ruta utilizable.

    Acepta tres formas: ninguna (usa el default de config), un nombre suelto
    ('marianela.txt' → data/raw/marianela.txt) o una ruta completa.
    Falla temprano y con contexto: un FileNotFoundError crudo de open() no
    te dice DÓNDE buscó ni qué alternativas había.
    """
    if indicado is None:
        return config.ARCHIVO_CORPUS

    # Si la ruta tal cual existe, la usamos. Si no, la buscamos en data/raw/.
    if indicado.exists():
        return indicado
    candidata = config.DIR_RAW / indicado.name
    if candidata.exists():
        return candidata

    disponibles = sorted(p.name for p in config.DIR_RAW.glob("*.txt"))
    raise FileNotFoundError(
        f"No encontré '{indicado}'. Corpus disponibles en data/raw/: {disponibles}"
    )


def palabras_muestra(tokens: list, n: int = 10) -> set[str]:
    """Elige automáticamente las palabras más frecuentes que SÍ se transforman.

    Una lista fija de palabras solo sirve para el texto con el que se escribió:
    con otro corpus la tabla sale vacía. Aquí seleccionamos las formas donde
    superficie y lema difieren —justo donde stemming y lematización divergen—
    ordenadas por frecuencia, para que la comparativa sea representativa.

    Counter.most_common(n) devuelve una lista de tuplas (elemento, frecuencia);
    la comprensión de conjunto se queda solo con el primer campo.
    """
    candidatas = Counter(
        t.text.lower() for t in tokens
        if t.text.lower() != limpiar_lema(t.lemma_.lower())
    )
    return {palabra for palabra, _ in candidatas.most_common(n)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalización y lematización (Módulo 3)"
    )
    parser.add_argument("--corpus", type=Path, default=None,
                        help="Ruta al .txt a procesar. Si es solo un nombre, se "
                             "busca en data/raw/. Por defecto: cap1_principito.txt")
    parser.add_argument("--guardar", action="store_true",
                        help="Escribe las tablas como CSV en outputs/<corpus>/")
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
    ruta = resolver_corpus(args.corpus)
    print(f"Corpus: {ruta.name}")
    texto = cargar_texto(ruta)
    print("Diagnóstico de carga:", diagnostico(texto))

    # --- 2. Pipeline ---
    doc = procesar(texto)
    print("Primeros 20 tokens:", [t.text for t in doc][:20])

    # --- 3. Lematización ---
    # tokens_de_contenido y lematizar aplican el MISMO filtro (es_contenido),
    # así que formas_contenido y lemas quedan alineados uno a uno. Si algún día
    # divergen, compresion_vocabulario lanza ValueError: la aserción está
    # dentro de la métrica, no confiada a que nos acordemos.
    tokens = tokens_de_contenido(doc)
    formas_contenido = [t.text for t in tokens]
    lemas, transformaciones = lematizar(doc)

    # Base honesta: excluimos puntuación y espacios del denominador. Contar los
    # saltos de línea del archivo como "tokens eliminados" infla la cifra.
    utiles = [t for t in doc if not (t.is_punct or t.is_space)]
    print(f"\nTokens totales (con puntuación y espacios): {len(doc)}")
    print(f"Tokens útiles (sin puntuación ni espacios):  {len(utiles)}")
    print(f"Tokens de contenido (tras filtrar stop words): {len(tokens)} "
          f"— filtrado elimina {1 - len(tokens)/len(utiles):.1%} del texto útil")
    print("Transformaciones (forma, POS, lema):", transformaciones[:10])

    # --- 4. Métricas, todas sobre la MISMA base ---
    muestra = palabras_muestra(tokens)

    df_compresion = compresion_vocabulario(formas_contenido, lemas)
    df_stem_lema = tabla_stem_vs_lema(doc, muestra)
    df_sobre = sobre_stemming(formas_contenido, lemas)

    print("\n=== Compresión de vocabulario (solo tokens de contenido) ===")
    print(df_compresion.to_string(index=False))

    print(f"\nType-Token Ratio: {type_token_ratio(formas_contenido):.3f} "
          f"— cerca de 1.0 significa poca redundancia léxica, y por tanto poco "
          f"margen de compresión para CUALQUIER técnica.")

    print("\n=== Stemming vs Lematización ===")
    print(df_stem_lema.to_string(index=False) if not df_stem_lema.empty
          else "Sin coincidencias en la muestra.")

    print("\n=== Over-stemming: dónde el stemmer fusionó de más ===")
    print(df_sobre.to_string(index=False) if not df_sobre.empty
          else "Sin casos en este corpus.")

    # --- 5. Corpus por oración (insumo del módulo 4) ---
    corpus = construir_corpus(doc)
    print(f"\nOraciones con contenido: {len(corpus)}")
    print("Primera oración lematizada:", corpus[0] if corpus else "—")

    # --- 6. Persistencia opcional ---
    if args.guardar:
        # Un subdirectorio por corpus: dos corridas no se pisan, y el nombre
        # del directorio documenta con qué dato se generó cada tabla.
        # .stem de un Path devuelve el nombre sin extensión.
        destino = config.DIR_OUTPUTS / ruta.stem
        destino.mkdir(parents=True, exist_ok=True)   # parents: crea intermedios
        df_compresion.to_csv(destino / "compresion.csv", index=False)
        df_stem_lema.to_csv(destino / "stem_vs_lema.csv", index=False)
        df_sobre.to_csv(destino / "over_stemming.csv", index=False)
        print(f"\nTablas guardadas en {destino}")


if __name__ == "__main__":     # solo ejecuta al invocar, no al importar
    main()
