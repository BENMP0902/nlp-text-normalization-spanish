"""
main.py — Orquestador / CLI. Punto de entrada único del pipeline.

Etapa 1 (módulo 3): normalización, lematización y métricas de vocabulario.
Etapa 2 (módulo 4): vectorización BoW / TF-IDF y proyección del espacio.
Etapa 3 (módulo 5): embeddings Word2Vec y proyección PCA del espacio léxico.

Las etapas se invocan por separado: --vectorizar y --embeddings no encadenan
con la etapa 1 porque operan sobre unidades distintas (un corpus vs. varios;
documentos vs. palabras) y mezclarlas produciría una salida de consola
imposible de leer.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np                          # requerido por np.savetxt en la etapa 2

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
from src.segmentacion import construir_documentos, segmentar_en_bloques
from src.vectorizacion import (
    construir_bow,
    construir_tfidf,
    diagnosticar,                           # nombre real de la función (no diagnosticar_idf)
    terminos_discriminantes,
    matriz_similitud,
    par_mas_similar,
    similitud_intra_vs_inter,
)
from src.visualizacion import proyectar, proyectar_pca, graficar
from src import embeddings as emb

# Directorio de figuras versionadas. Se resuelve desde la ubicación de este
# archivo y no desde el cwd: `python main.py` desde otro directorio no rompe.
# Distinción frente a outputs/: docs/ es documentación referenciada por el
# README y se versiona; outputs/ son artefactos regenerables y no se versiona.
DIR_DOCS = Path(__file__).resolve().parent / "docs"


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


# =====================================================================
# ETAPA 2 — Vectorización (módulo 4)
# =====================================================================

def _lematizar_cada(documentos: list[str]) -> list[list[str]]:
    """Lematiza cada documento de texto por separado."""
    # Una llamada a spaCy por documento. Aceptable con estos corpus; para
    # volumen real, nlp.pipe(textos, batch_size=...) procesa en lote y evita
    # el overhead de reconstruir el pipeline en cada invocación.
    # lematizar() devuelve (lemas, transformaciones); aquí solo interesa [0].
    return [lematizar(procesar(d))[0] for d in documentos]


# Las tres estrategias comparten firma (texto, oraciones_por_bloque) para que
# el diccionario de despacho pueda invocarlas sin distinguir casos. Las que no
# usan oraciones_por_bloque lo reciben y lo ignoran.

def _por_oracion(texto: str, oraciones_por_bloque: int) -> list[list[str]]:
    """Una oración = un documento. Reproduce el material de referencia.

    construir_corpus() ya lo implementa desde el módulo 3 y lematiza cada
    oración dentro del contexto del texto completo. Con documentos tan cortos,
    df≈1 para casi todo el vocabulario, el IDF pierde varianza y TF-IDF
    degenera en BoW binario L2-normalizado.
    """
    return construir_corpus(procesar(texto))


def _por_parrafo(texto: str, oraciones_por_bloque: int) -> list[list[str]]:
    """Un párrafo = un documento: masa suficiente para que el IDF discrimine.

    `procesar` se pasa como el callable nlp. construir_documentos solo lo usa
    en la ruta de respaldo, cuando el .txt no trae párrafos separados por
    línea en blanco; oraciones_por_bloque aplica también solo ahí.
    """
    return _lematizar_cada(
        construir_documentos(texto, procesar, oraciones_por_bloque=oraciones_por_bloque)
    )


def _por_bloque(texto: str, oraciones_por_bloque: int) -> list[list[str]]:
    """N oraciones consecutivas = un documento.

    Es la estrategia de respaldo de _por_parrafo expuesta directamente, para
    textos sin párrafos (líneas duras cada 80 columnas). No es una unidad
    lingüística: el tamaño del bloque es arbitrario.
    """
    return _lematizar_cada(segmentar_en_bloques(procesar(texto), oraciones_por_bloque))


# Diccionario de despacho: ÚNICA fuente de verdad de las granularidades.
# Los choices de --granularidad se construyen desde sus claves. Antes eran dos
# listas escritas a mano en sitios distintos y divergieron ("bloque" en la CLI,
# "parrafo" en el código) sin que nada fallara.
ESTRATEGIAS = {
    "oracion": _por_oracion,
    "parrafo": _por_parrafo,
    "bloque": _por_bloque,
}


def documentos_lematizados(texto: str, granularidad: str,
                           oraciones_por_bloque: int = 4) -> list[list[str]]:
    """Aplica la etapa 1 sobre el texto y devuelve documentos como listas de lemas.

    La granularidad es el PARÁMETRO DEL EXPERIMENTO, no un detalle interno.
    Exponer las rutas convierte una afirmación del README en una medición. Es
    el mismo patrón de --sin-negadores: reproducir el comportamiento del
    material y cuantificar la diferencia.
    """
    # Validación explícita: argparse ya filtra la CLI, pero esta función
    # también se invoca desde código y tests. Un valor desconocido no debe
    # caer en silencio a ninguna estrategia.
    if granularidad not in ESTRATEGIAS:
        raise ValueError(
            f"Granularidad desconocida: {granularidad!r}. "
            f"Valores válidos: {sorted(ESTRATEGIAS)}"
        )
    return ESTRATEGIAS[granularidad](texto, oraciones_por_bloque)


def sufijo_configuracion(granularidad: str, oraciones_por_bloque: int) -> str:
    """Identificador de la configuración para nombres de archivo y títulos.

    Con 'bloque', N cambia los documentos y por tanto todos los resultados:
    sin N en el nombre, una corrida con 8 sobrescribía la de 4 sin aviso.
    Las otras granularidades no dependen de N (salvo el respaldo automático
    de 'parrafo', que se decide por corpus en tiempo de ejecución).
    """
    if granularidad == "bloque":
        return f"bloque{oraciones_por_bloque}"
    return granularidad


def ejecutar_vectorizacion(rutas_corpus: list[Path],
                           granularidad: str = "parrafo",
                           guardar: bool = False,
                           oraciones_por_bloque: int = 4) -> None:
    """Construye el espacio vectorial sobre todos los corpus y lo diagnostica."""
    corpus_tokens: list[list[str]] = []
    etiquetas: list[str] = []

    for ruta in rutas_corpus:
        # cargar_texto() y no read_text(): valida encoding y detecta mojibake.
        # La doble codificación es UTF-8 válido y no lanza ninguna excepción.
        texto = cargar_texto(ruta)
        for tokens in documentos_lematizados(texto, granularidad, oraciones_por_bloque):
            if tokens:                       # descarta documentos vacíos tras el filtrado
                corpus_tokens.append(tokens)
                etiquetas.append(ruta.stem)  # "cap1_principito" → etiqueta de origen

    # Guarda de entrada: con menos de 2 documentos el IDF es indefinido.
    # Fallar aquí es preferible a producir una matriz sin significado.
    if len(corpus_tokens) < 2:
        raise ValueError(
            f"Solo {len(corpus_tokens)} documento(s). La etapa 2 requiere al menos 2; "
            f"agrega corpus a data/raw/ o baja min_caracteres en segmentacion.py"
        )

    print(f"\n=== Etapa 2: vectorización (granularidad: {granularidad}) ===")
    print(f"Documentos: {len(corpus_tokens)} | Corpus: {sorted(set(etiquetas))}")

    X_bow, v_bow = construir_bow(corpus_tokens)
    X_tfidf, v_tfidf = construir_tfidf(corpus_tokens)

    print(f"BoW      {diagnosticar(X_bow, v_bow)}")
    print(f"TF-IDF   {diagnosticar(X_tfidf, v_tfidf)}")
    print("  ↑ idf_std < 0.25 significa que el IDF no discrimina: documentos "
          "demasiado cortos o demasiado pocos.")

    # --- Prueba cuantitativa de la hipótesis del experimento ---
    S = matriz_similitud(X_tfidf)
    i, j, s = par_mas_similar(S)
    print(f"\nPar más similar: doc {i} ({etiquetas[i]}) ↔ doc {j} ({etiquetas[j]}) | cos={s:.3f}")
    print(f"Similitud media: {similitud_intra_vs_inter(S, etiquetas)}")
    print("  ↑ si intra_corpus no supera claramente a inter_corpus, la "
          "vectorización no capturó tema y el gráfico no debe interpretarse.")

    tabla = terminos_discriminantes(X_tfidf, v_tfidf, etiquetas)
    print("\n=== Términos más discriminantes (top 5 por corpus) ===")
    print(tabla[tabla["rango"] <= 5].to_string(index=False))

    coords, var = proyectar(X_tfidf)
    print(f"\nVarianza explicada por componente: {var.round(4)} "
          f"| acumulada: {var.sum():.1%}")

    if guardar:
        # El nombre de archivo lleva la configuración: dos corridas no se pisan
        # y el nombre documenta con qué configuración se generó cada artefacto.
        sufijo = sufijo_configuracion(granularidad, oraciones_por_bloque)
        destino = config.DIR_OUTPUTS / "vectorizacion"
        destino.mkdir(parents=True, exist_ok=True)
        tabla.to_csv(destino / f"terminos_{sufijo}.csv",
                     index=False, encoding="utf-8")
        np.savetxt(destino / f"similitud_{sufijo}.csv",
                   S, delimiter=",", fmt="%.4f")

        figura = graficar(
            coords, etiquetas, var,
            f"Espacio TF-IDF proyectado (LSA) — granularidad: {sufijo}",
            DIR_DOCS / f"proyeccion_lsa_{sufijo}.png",
        )
        print(f"\nTablas en {destino} · figura en {figura}")
    else:
        print("\n(sin --guardar: no se escribieron figura ni CSV)")


# =====================================================================
# ETAPA 3 — Embeddings Word2Vec (módulo 5)
# =====================================================================

# Palabras graficadas en la figura de frecuencia y sondas del diagnóstico.
N_FRECUENTES = 60
N_SONDAS = 6


def ejecutar_embeddings(rutas_corpus: list[Path], guardar: bool = False) -> None:
    """Entrena Word2Vec sobre los corpus, diagnostica su suficiencia y grafica."""
    if aviso := emb.verificar_semilla_hash():
        print(aviso)

    oraciones: list[list[str]] = []
    pos_por_lema: dict = {}
    for ruta in rutas_corpus:
        # cargar_texto() valida encoding y mojibake igual que en las etapas 1 y 2.
        o, pos = emb.preparar_corpus(cargar_texto(ruta))
        oraciones.extend(o)
        for lema, conteo in pos.items():
            pos_por_lema.setdefault(lema, Counter()).update(conteo)
        print(f"[embeddings] {ruta.name}: {len(o)} oraciones, "
              f"{sum(map(len, o)):,} tokens")

    modelo = emb.entrenar(oraciones)
    contenido = emb.palabras_de_contenido(modelo, pos_por_lema)
    if len(contenido) < N_SONDAS:
        raise ValueError(
            f"Solo {len(contenido)} palabras de contenido superan min_count="
            f"{config.W2V_MIN_COUNT}. El corpus no alcanza para la etapa 3."
        )
    sondas = contenido[:N_SONDAS]

    print("\n=== Etapa 3: embeddings Word2Vec ===")
    print(f"Corpus: {[r.name for r in rutas_corpus]}")
    print(f"Hiperparámetros: dim={config.W2V_VECTOR_SIZE} ventana={config.W2V_VENTANA} "
          f"min_count={config.W2V_MIN_COUNT} sample={config.W2V_SAMPLE} "
          f"sg={config.W2V_SG} epochs={config.W2V_EPOCHS} seed={config.W2V_SEMILLA} "
          f"workers={config.W2V_WORKERS}")

    diag = emb.diagnosticar_suficiencia(modelo, oraciones, sondas)
    avisos = diag.pop("avisos")
    print(f"Diagnóstico: {diag}")
    for a in avisos:
        print(a)

    print(f"\n=== Vecinos más cercanos (top {config.W2V_TOP_K}, coseno) ===")
    for s in sondas:
        vecinos = modelo.wv.most_similar(s, topn=config.W2V_TOP_K)
        print(f"{s:>12} → " + ", ".join(f"{w} ({c:.4f})" for w, c in vecinos))
    if avisos:
        print("  ↑ con los avisos de arriba activos, estos decimales NO son "
              "evidencia: la lista cambia con la semilla.")

    # --- Proyecciones (sin stop words: ver stop_words_para_graficar) ---
    frecuentes = contenido[:N_FRECUENTES]
    coords_f, var_f = proyectar_pca(emb.matriz_de(modelo, frecuentes))
    palabras_v, grupos_v = emb.vecindarios(modelo, sondas, set(contenido))
    coords_v, var_v = proyectar_pca(emb.matriz_de(modelo, palabras_v))
    print(f"\nVarianza explicada — frecuentes: {var_f.round(4)} (acum. {var_f.sum():.1%}) "
          f"| vecindarios: {var_v.round(4)} (acum. {var_v.sum():.1%})")

    if guardar:
        destino = config.DIR_OUTPUTS / "embeddings"
        destino.mkdir(parents=True, exist_ok=True)
        modelo.save(str(destino / "word2vec.model"))     # gensim exige str
        texto = emb.guardar_texto_procesado(
            oraciones, config.DIR_PROCESADO / "corpus_embeddings.txt")

        f1 = graficar(
            coords_f, [emb.pos_dominante(w, pos_por_lema) for w in frecuentes], var_f,
            f"Word2Vec — {len(frecuentes)} palabras de contenido más frecuentes (PCA)\n"
            f"varianza relativa a las palabras graficadas, no al vocabulario",
            DIR_DOCS / "embeddings_frecuentes.png",
            anotaciones=frecuentes, titulo_leyenda="POS dominante",
            prefijo_eje="Componente principal",
        )
        f2 = graficar(
            coords_v, grupos_v, var_v,
            f"Word2Vec — vecindarios de {len(sondas)} sondas, "
            f"top {config.W2V_TOP_K} vecinos de contenido (PCA)\n"
            f"varianza relativa a las palabras graficadas, no al vocabulario",
            DIR_DOCS / "embeddings_vecindarios.png",
            anotaciones=palabras_v, titulo_leyenda="Sonda",
            prefijo_eje="Componente principal",
        )
        print(f"\nModelo en {destino} · texto procesado en {texto} · "
              f"figuras en {f1} y {f2}")
    else:
        print("\n(sin --guardar: no se escribieron modelo, texto ni figuras)")


def rutas_de_etapa(corpus: Path | None, etapa: str) -> list[Path]:
    """Corpus de las etapas multi-corpus: todos los de data/raw/ o solo --corpus."""
    if corpus is not None:
        print(f"[aviso] {etapa} con un solo corpus: hay menos texto sobre el cual "
              "estimar. Omite --corpus para usar todos los de data/raw/.")
        return [resolver_corpus(corpus)]
    rutas = sorted(config.DIR_RAW.glob("*.txt"))
    if not rutas:
        raise FileNotFoundError(f"No hay archivos .txt en {config.DIR_RAW}")
    return rutas


# =====================================================================
# CLI
# =====================================================================

def construir_parser() -> argparse.ArgumentParser:
    """Define la CLI. Separada de main() para que los tests puedan inspeccionarla."""
    parser = argparse.ArgumentParser(
        description="Pipeline de normalización (etapa 1), vectorización (etapa 2) "
                    "y embeddings (etapa 3)"
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
    parser.add_argument("--vectorizar", action="store_true",
                        help="Ejecuta la etapa 2 (feature extraction) sobre data/raw/")
    parser.add_argument("--embeddings", action="store_true",
                        help="Ejecuta la etapa 3 (Word2Vec) sobre data/raw/. "
                             "Para reproducibilidad: PYTHONHASHSEED=0 python main.py --embeddings")
    # choices sale de ESTRATEGIAS: la CLI no puede ofrecer un valor que el
    # código no sepa despachar, ni al revés.
    parser.add_argument("--granularidad", choices=list(ESTRATEGIAS), default="parrafo",
                        help="Unidad que constituye un documento en la etapa 2. "
                             "'parrafo' (default) delimita por líneas en blanco; "
                             "'oracion' reproduce el material de referencia; "
                             "'bloque' agrupa N oraciones consecutivas y es la "
                             "estrategia de respaldo para textos sin párrafos, "
                             "no una tercera unidad lingüística.")
    parser.add_argument("--oraciones-por-bloque", type=int, default=4,
                        help="Oraciones por documento con --granularidad=bloque "
                             "(y en el respaldo automático de 'parrafo'). Default: 4")
    return parser


def main() -> None:
    args = construir_parser().parse_args()

    # Sobrescribimos config desde la CLI: permite comparar variantes sin editar
    # código. argparse convierte --sin-negadores en args.sin_negadores (guion
    # bajo), porque los guiones no son válidos en identificadores de Python.
    # Va ANTES de cualquier procesamiento: procesar() lee config al ejecutarse,
    # así que ambas etapas heredan la configuración elegida.
    if args.sin_negadores:
        config.PRESERVAR_NEGADORES = False
    if args.sin_filtro:
        config.FILTRAR_STOPWORDS = False

    if args.vectorizar and args.embeddings:
        raise SystemExit("--vectorizar y --embeddings son etapas separadas: "
                         "invócalas en corridas distintas.")

    # --- Etapa 3: ruta alterna con retorno temprano ---
    if args.embeddings:
        ejecutar_embeddings(rutas_de_etapa(args.corpus, "--embeddings"),
                            guardar=args.guardar)
        return   # la etapa 3 no encadena con ninguna otra

    # --- Etapa 2: ruta alterna con retorno temprano ---
    if args.vectorizar:
        # --corpus restringe la etapa 2 a un solo texto. Se permite, pero con
        # un corpus el IDF tiene mucho menos sobre qué discriminar.
        rutas = rutas_de_etapa(args.corpus, "--vectorizar")
        ejecutar_vectorizacion(rutas,
                               granularidad=args.granularidad,
                               guardar=args.guardar,
                               oraciones_por_bloque=args.oraciones_por_bloque)
        return   # la etapa 2 no encadena con la etapa 1

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

    # --- 5. Corpus por oración (insumo de la etapa 2) ---
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
