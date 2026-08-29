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
