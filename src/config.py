"""Configuración central: todas las decisiones de diseño del pipeline.

Regla: si un parámetro cambia el RESULTADO del preprocesamiento, vive aquí.
Motivo: en PLN, la mayor fuente de irreproducibilidad no es el modelo, es el
preprocesamiento. Si tus negadores preservados y tu modelo viven dispersos por
el código, nadie —ni tú en tres meses— puede reproducir tu resultado.
"""
from pathlib import Path

# --- Rutas ------------------------------------------------------------------
# __file__ es la ruta de ESTE archivo. .resolve() la vuelve absoluta (resuelve
# symlinks y '..'). parents[1] sube un nivel: de src/ a la raíz del proyecto.
# Ventaja: el script funciona sin importar desde qué directorio lo invoques.
RAIZ = Path(__file__).resolve().parents[1]

DIR_RAW = RAIZ / "data" / "raw"          # el operador / de pathlib une rutas
DIR_PROCESADO = RAIZ / "data" / "processed"
DIR_OUTPUTS = RAIZ / "outputs"

# Corpus POR DEFECTO. La CLI puede sobreescribirlo con --corpus.
# Es un valor por defecto, no una decisión de diseño del sistema.
ARCHIVO_CORPUS = DIR_RAW / "cap1_principito.txt"

# --- Modelo -----------------------------------------------------------------
MODELO_SPACY = "es_core_news_sm"

# Desactivamos NER: no extraemos entidades y cuesta cómputo sin uso.
# NO desactivamos 'parser': doc.sents lo necesita para segmentar oraciones,
# y el módulo 4 depende de esa segmentación.
# Tupla, no lista: debe ser hashable para servir de clave en lru_cache.
COMPONENTES_DESACTIVADOS = ("ner",)

# --- Decisiones de normalización --------------------------------------------
# Negadores EXCLUIDOS del filtrado de stop words.
# Justificación lingüística: en español la negación es una partícula
# independiente, sin redundancia morfológica que la recupere. Si la eliminas,
# "no falló" y "falló" colapsan al mismo vector con polaridad opuesta.
NEGADORES = frozenset({
    "no", "ni", "nada", "nunca", "jamás", "jamas",
    "tampoco", "sin", "nadie", "ningún", "ninguna", "ninguno",
})

FILTRAR_STOPWORDS = True       # conmutable para comparar resultados
PRESERVAR_NEGADORES = True     # si False, reproduces el comportamiento del curso

# --- Etapa 3: embeddings Word2Vec (módulo 5) ---------------------------------
# Los hiperparámetros cambian los vectores, así que viven aquí y no en la CLI.
W2V_VECTOR_SIZE = 100      # dimensiones; con ~60 k tokens, más dimensiones
                           # solo añaden parámetros que el corpus no puede estimar
W2V_VENTANA = 5            # palabras de contexto a cada lado
W2V_MIN_COUNT = 5          # por debajo, el vector de una palabra es casi su
                           # inicialización aleatoria: menos de 5 contextos
W2V_SAMPLE = 1e-3          # submuestreo de palabras frecuentes (Mikolov et al.,
                           # 2013). Sustituye al filtrado de stop words: las
                           # degrada sin borrarlas de la ventana
W2V_SG = 1                 # skip-gram: aprende mejor que CBOW las palabras poco
                           # frecuentes, que en un corpus chico son casi todas
W2V_EPOCHS = 30            # el default (5) está pensado para millones de tokens;
                           # un libro necesita más pasadas para converger
W2V_SEMILLA = 42
W2V_WORKERS = 1            # con varios hilos el orden de las actualizaciones no
                           # es determinista y la semilla deja de bastar
                           # (documentación de gensim, Word2Vec, parámetro seed)

# Umbrales de suficiencia: por debajo, el código ADVIERTE en vez de callar.
# most_similar devuelve vecinos con cuatro decimales aunque sean ruido.
W2V_MIN_TOKENS = 100_000   # heurística de este proyecto, no un estándar: el
                           # artículo original entrenó con miles de millones
W2V_MIN_ESTABILIDAD = 0.5  # Jaccard medio del top-k de vecinos entre dos
                           # semillas. Si los vecinos cambian con la semilla,
                           # no son propiedad del corpus sino de la inicialización
W2V_TOP_K = 10             # vecinos por sonda en el diagnóstico y las figuras
