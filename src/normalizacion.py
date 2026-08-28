"""Pipeline de normalización: carga del modelo, filtrado y lematización."""
from functools import lru_cache

import spacy
from spacy.language import Language
from spacy.tokens import Doc

from src import config


@lru_cache(maxsize=1)
def cargar_modelo(nombre: str, desactivar: tuple) -> Language:
    """Carga el modelo de spaCy una sola vez por proceso.

    lru_cache memoiza la llamada: la segunda invocación con los MISMOS
    argumentos devuelve el objeto ya en memoria en vez de releer ~13 MB de
    disco y reconstruir el pipeline. Por eso los argumentos deben ser
    hashables — de ahí que COMPONENTES_DESACTIVADOS sea tupla y no lista.
    Una lista lanzaría TypeError: unhashable type.
    """
    try:
        # disable= excluye componentes del pipeline SIN cargarlos.
        # Distinto de exclude=, que además impide reactivarlos después.
        return spacy.load(nombre, disable=list(desactivar))
    except OSError as e:
        # NO descargamos en runtime. Fallamos con instrucción explícita.
        # `spacy.cli.download()` dentro de un except es el patrón del curso:
        # funciona en un notebook, mata tu build en CI o en un contenedor
        # sin red, y encima instala una versión no fijada.
        raise OSError(
            f"Modelo '{nombre}' no instalado. Ejecuta:\n"
            f"    pip install -r requirements.txt\n"
            f"Error original: {e}"
        ) from e            # `from e` preserva la cadena de excepciones


def stop_words_efectivas(nlp: Language) -> frozenset:
    """Lista de stop words que realmente vamos a aplicar.

    Diferencia de conjuntos: a la lista de spaCy le restamos los negadores.
    Con PRESERVAR_NEGADORES=False reproduces el pipeline del curso, lo que
    permite comparar ambos comportamientos empíricamente en vez de discutirlos.
    """
    base = frozenset(nlp.Defaults.stop_words)
    if config.PRESERVAR_NEGADORES:
        return base - config.NEGADORES      # `-` sobre frozenset = diferencia
    return base


def es_contenido(token, stop_words: frozenset) -> bool:
    """Decide si un token aporta contenido semántico.

    El punto fino está en el DOBLE nivel de verificación de stop words.
    spaCy evalúa `token.is_stop` sobre la FORMA SUPERFICIAL. Una conjugación
    de un verbo-stopword puede pasar el filtro y luego lematizarse a un lema
    que sí estaba en la lista → vocabulario inconsistente entre corridas.
    Verificamos forma Y lema.
    """
    if token.is_punct or token.is_space:
        return False
    # like_num cubre "3" y "tres"; like_url y like_email evitan que un enlace
    # entre al vocabulario como token único e irrepetible.
    if token.like_num or token.like_url or token.like_email:
        return False

    if config.FILTRAR_STOPWORDS:
        lema = token.lemma_.lower()
        if token.lower_ in stop_words or lema in stop_words:
            return False

    return True


def procesar(texto: str) -> Doc:
    """Ejecuta el pipeline de spaCy sobre el texto completo."""
    nlp = cargar_modelo(config.MODELO_SPACY, config.COMPONENTES_DESACTIVADOS)

    # nlp.max_length (1,000,000 por defecto) existe porque el parser tiene
    # consumo de memoria cuadrático en el peor caso. Un capítulo no lo alcanza;
    # el libro completo podría. Fallar aquí es mejor que un OOM kill del SO.
    if len(texto) > nlp.max_length:
        raise ValueError(
            f"Texto de {len(texto)} chars supera nlp.max_length={nlp.max_length}. "
            "Procesa por capítulos con nlp.pipe()."
        )

    return nlp(texto)


def lematizar(doc: Doc) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Extrae lemas de contenido en UNA sola pasada sobre el doc.

    Devuelve:
      - lemas: lista de lemas en minúsculas
      - transformaciones: (forma_original, POS, lema) donde la forma cambió.
        Es tu material de inspección: si esta lista viene casi vacía, tu
        lematizador está en modo lookup y no está haciendo análisis morfológico.

    ORDEN CRÍTICO: lematizar primero, bajar a minúsculas DESPUÉS.
    Al revés destruyes la señal de mayúscula que usa el POS tagger, y el
    lematizador de reglas depende del POS. Error frecuente y silencioso:
    no lanza excepción, solo degrada la calidad de los lemas.
    """
    nlp = cargar_modelo(config.MODELO_SPACY, config.COMPONENTES_DESACTIVADOS)
    stop_words = stop_words_efectivas(nlp)

    lemas: list[str] = []
    transformaciones: list[tuple[str, str, str]] = []

    for token in doc:
        if not es_contenido(token, stop_words):
            continue
        lema =  limpiar_lema(token.lemma_.lower())
        lemas.append(lema)
        if token.lower_ != lema:
            transformaciones.append((token.text, token.pos_, lema))

    return lemas, transformaciones


def construir_corpus(doc: Doc) -> list[list[str]]:
    """Segmenta en oraciones y lematiza cada una: el insumo del módulo 4.

    doc.sents requiere el parser activo. Si desactivaras 'parser' en config,
    esto lanzaría ValueError — de ahí la advertencia en COMPONENTES_DESACTIVADOS.

    Devolvemos lista de LISTAS, no de strings, para que el módulo 4 pase las
    oraciones directo al vectorizador sin re-tokenizar. Re-tokenizar con el
    tokenizador por defecto de scikit-learn después de haber usado spaCy es
    incoherencia de pipeline: dos criterios distintos sobre el mismo texto.
    """
    nlp = cargar_modelo(config.MODELO_SPACY, config.COMPONENTES_DESACTIVADOS)
    stop_words = stop_words_efectivas(nlp)

    corpus = []
    for oracion in doc.sents:
        lemas = [limpiar_lema(t.lemma_.lower()) for t in oracion if es_contenido(t, stop_words)]
        if lemas:                   # descarta oraciones vacías tras el filtrado
            corpus.append(lemas)
    return corpus

def limpiar_lema(lema: str) -> str:
    """Descarta el pronombre enclítico que spaCy adjunta al lema del verbo.

    En español el clítico se suelda al verbo ('masticarla', 'dárselo'). spaCy
    devuelve 'masticar él' — dos palabras en un solo lema. Nos quedamos con la
    primera: el contenido semántico está en el verbo, y el pronombre sería
    stop word de todos modos.

    str.split() sin argumentos separa por cualquier secuencia de espacios y
    descarta los vacíos; [0] toma el primer elemento. Sobre un lema de una sola
    palabra devuelve esa palabra sin cambios, así que es seguro aplicarlo
    siempre y no hace falta condicional.
    """
    return lema.split()[0]


def tokens_de_contenido(doc: Doc) -> list:
    """Los tokens que sobreviven al filtrado. Fuente única de verdad.

    Existe para que formas, stems y lemas se calculen sobre EL MISMO conjunto.
    Compararlos sobre conjuntos distintos produce una tabla que parece válida
    y no lo es — que es exactamente el error que esta función previene.
    """
    nlp = cargar_modelo(config.MODELO_SPACY, config.COMPONENTES_DESACTIVADOS)
    stop_words = stop_words_efectivas(nlp)
    return [t for t in doc if es_contenido(t, stop_words)]
