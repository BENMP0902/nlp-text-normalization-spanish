# nlp-text-normalization-spanish

Pipeline reproducible de **normalización y lematización de texto en español** con spaCy, construido para evaluar empíricamente el efecto de cada etapa del preprocesamiento sobre el vocabulario resultante.

El proyecto no se limita a ejecutar el pipeline: **mide** su efecto, **documenta** sus fallos y **verifica** con tests que las decisiones de diseño se cumplen.

Trabajo académico de la asignatura de Procesamiento de Lenguaje Natural — Ingeniería en Software, Hybridge Education.

---

## Tesis del proyecto

Un pipeline de normalización mal calibrado destruye información sin avisar. Este repositorio demuestra dos casos concretos con datos:

1. **El filtrado de stop words elimina los negadores**, y con ellos la polaridad de la oración. `"no falló"` y `"falló"` producen vectores idénticos.
2. **La compresión de vocabulario no mide la calidad de la normalización**, sino la redundancia léxica del corpus. Reportar un porcentaje sin la Type-Token Ratio que lo contextualiza es reportar un número sin significado.

---

## Resultados

Dos corpus procesados con el mismo pipeline y la misma configuración.

### Métricas generales

| | *El Principito*, cap. 1 | *Caperucita Roja* |
|---|---|---|
| Caracteres | 2,423 | 6,078 |
| Tokens totales (con puntuación y espacios) | 516 | 1,372 |
| Tokens útiles (sin puntuación ni espacios) | 422 | 1,089 |
| Tokens de contenido (tras filtrar stop words) | 166 | 434 |
| Filtrado sobre texto útil | 60.7 % | 60.1 % |
| Oraciones con contenido | 27 | 63 |

### Compresión de vocabulario

Las tres cifras se calculan sobre **el mismo conjunto de tokens** — los de contenido. Comparar formas sin filtrar contra lemas filtrados mezcla el efecto del filtrado con el de la normalización y produce una tabla inválida.

| Tratamiento | \|V\| Principito | Reducción | \|V\| Caperucita | Reducción |
|---|---|---|---|---|
| Sin normalizar | 129 | — | 265 | — |
| Stemming (Snowball) | 120 | 7.0 % | 239 | 9.8 % |
| Lematización (spaCy) | 125 | 3.1 % | 248 | 6.4 % |
| **Type-Token Ratio** | **0.777** | | **0.611** | |

**Interpretación.** La lematización comprime poco en ambos corpus, y no es un defecto del pipeline: con una TTR de 0.777, el 78 % de las palabras de contenido de *El Principito* aparecen una sola vez (*hapax legomena*). No hay variantes que fusionar. Al pasar a un corpus con más redundancia léxica —TTR 0.611— la compresión por lematización se duplica.

Esto sostiene la tesis de que **la lematización no es una técnica de compresión sino de generalización**. Su valor aparece frente a texto no visto: si el corpus de entrenamiento contiene `tragaba` y un documento nuevo dice `tragarán`, sin lematización son términos sin relación y el vector pierde la señal. Medir eso requiere cobertura *out-of-vocabulary* sobre un conjunto de prueba, no `|V|` sobre un solo texto.

### Efecto de preservar los negadores

| | Principito | Caperucita |
|---|---|---|
| Tokens de contenido, con negadores | 166 | 434 |
| Tokens de contenido, sin negadores | 161 | 406 |
| **Diferencia** | 5 (3.0 %) | **28 (6.5 %)** |
| \|V\| lematizado, con / sin | 125 / 122 | 248 / 243 |
| TTR, con / sin | 0.777 / — | 0.611 / 0.640 |

La lista de stop words de spaCy para español incluye `no`, `ni`, `nada`, `nunca`, `sin`, `tampoco`, `nadie` y `ningún`. En español la negación es una **partícula independiente sin redundancia morfológica**: nada más en la oración la recupera. Eliminarla no reduce ruido, invierte el significado.

Ejemplo tomado de *Caperucita Roja*. Oración original:

> «Había una vez una adorable niña que era querida por todo aquél que la conociera, pero sobre todo por su abuelita, y **no** quedaba **nada** que **no** hubiera hecho por la niña.»

Salida del pipeline en ambas configuraciones:

```
con negadores:  ['adorable','niña','querer','conociero','abuelita','no','quedar','nada','no','niña']
sin negadores:  ['adorable','niña','querer','conociero','abuelita',     'quedar',       'niña']
```

La construcción es una **doble negación** que en español afirma: la abuela hizo absolutamente todo por la niña. La versión filtrada deja `quedar` a secas y elimina una estructura lógica de dos niveles. Ningún modelo entrenado sobre esos vectores puede recuperarla, porque la información ya no está.

Reproducir el comportamiento por defecto del material de referencia:

```bash
python main.py --sin-negadores
```

---

## Limitaciones documentadas del modelo

Hallazgos observados al comparar las dos corridas. Se reportan porque son **sistemáticos y predecibles**, no ruido aleatorio.

### Lematizador (`es_core_news_sm` 3.8.0)

| Fenómeno | Ejemplos | Consecuencia |
|---|---|---|
| **Lemas inexistentes** | `conociera` → `conociero`; `criatura` → `criaturar` | Términos que no son palabras entran al vocabulario |
| **Masculinización de femeninos** | `caperuza` → `caperuzo`; `abuelita` → `abuelito`; `roja` → `rojo` | Sustantivos femeninos etiquetados como ADJ reciben lematización de adjetivo |
| **Inconsistencia intra-corpus** | `abuelita` → `{abuelita, abuelito}`; `criatura` → `{criatura, criaturar}` | La misma forma superficial recibe lemas distintos según el POS asignado en cada contexto: la normalización **divide** un término en vez de unificarlo |
| **Pretérito de 1ª persona** | `viví`, `dibujé`, `abandoné` sin transformar | Formas acentuadas de pretérito perfecto simple no se reducen al infinitivo |

El origen de la mayoría de estos casos es un **error de etiquetado POS** que se propaga al lema: el lematizador de reglas depende del POS tagger, así que una etiqueta equivocada produce un lema equivocado sin lanzar ninguna excepción.

### Stemmer (Snowball / Porter2 español)

**Under-stemming en palabras cortas.** `años` → `años` y `ojos` → `ojos` conservan la marca de plural, mientras que `flores` → `flor` y `personas` → `person` sí se procesan. El patrón común de los fallos es la forma vocal–consonante–vocal–`s`: el algoritmo solo elimina sufijos que caen dentro de la región **RV**, calculada a partir de las primeras letras, y en ese patrón RV empieza demasiado tarde para alcanzar la `-s`.

### Sobre la tabla de divergencia

`sobre_stemming()` agrupa por stem y muestra los grupos con más de un lema. **Revela desacuerdo, no culpabilidad.** Un grupo puede originarse en over-stemming (el stemmer fusionó de más) o en un fallo del lematizador (no redujo lo que debía). En *El Principito*, 4 de 5 grupos correspondían a fallos del lematizador, no del stemmer.

Ninguna de las dos técnicas es *ground truth* por sí sola. Distinguir el origen de cada divergencia exige inspección manual.

---

## Estructura

```
.
├── main.py                      # orquestador / CLI — punto de entrada único
├── data/
│   ├── raw/                     # corpus de entrada (inmutable: solo lectura)
│   └── processed/               # artefactos derivados
├── src/
│   ├── config.py                # decisiones de diseño, centralizadas
│   ├── carga.py                 # I/O + saneamiento de encoding
│   ├── normalizacion.py         # filtrado, lematización, segmentación
│   └── metricas.py              # compresión, TTR, divergencia stem/lema
├── tests/
│   └── test_normalizacion.py    # 6 tests
├── outputs/                     # tablas CSV (no versionadas)
├── requirements.txt             # dependencias directas (intención)
├── requirements.lock.txt        # árbol completo resuelto (estado)
└── .python-version              # intérprete fijado vía pyenv
```

**Separación config / invocación.** `config.py` contiene los parámetros que definen el comportamiento del sistema (modelo, negadores preservados, componentes desactivados); se versionan porque cambian el resultado. La ruta del corpus es un parámetro de **invocación**, no de diseño, y por eso vive en la CLI: procesar otro texto no debe requerir editar código fuente.

---

## Requisitos

- Python 3.12.12 (fijado en `.python-version`, gestionado con [pyenv](https://github.com/pyenv/pyenv))
- spaCy 3.8.16 · es_core_news_sm 3.8.0 · NLTK 3.9.1 · scikit-learn 1.5.2 · pandas 2.2.3 · pytest 8.3.3

### Instalación

```bash
git clone git@github.com:BENMP0902/nlp-text-normalization-spanish.git
cd nlp-text-normalization-spanish

python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m spacy validate           # verifica compatibilidad modelo ↔ librería
pytest                             # esperado: 6 passed
```

El modelo se instala **fijado por URL del release**, no con `spacy download` en tiempo de ejecución. Descargar en runtime rompe cualquier build sin salida a internet (contenedor, CI) e instala una versión no fijada.

---

## Uso

```bash
python main.py                                        # corpus por defecto
python main.py --corpus caperucita_roja.txt           # otro corpus de data/raw/
python main.py --corpus /ruta/absoluta/texto.txt      # ruta completa
python main.py --corpus mi_texto.txt --guardar        # exporta CSV a outputs/<corpus>/
python main.py --sin-negadores                        # filtra también los negadores
python main.py --sin-filtro                           # sin filtrado de stop words
```

| Bandera | Efecto |
|---|---|
| `--corpus RUTA` | Texto a procesar. Un nombre suelto se busca en `data/raw/`. |
| `--guardar` | Escribe `compresion.csv`, `stem_vs_lema.csv` y `over_stemming.csv` en `outputs/<nombre_corpus>/` |
| `--sin-negadores` | Desactiva la preservación de negadores |
| `--sin-filtro` | Desactiva por completo el filtrado de stop words |

---

## Agregar tus propios corpus

El pipeline está parametrizado: **no hay que tocar código para procesar un texto nuevo.**

```bash
# 1. Coloca el .txt en data/raw/ (la barra final evita renombrados accidentales)
cp /ruta/a/tu_texto.txt data/raw/

# 2. Ejecuta
python main.py --corpus tu_texto.txt --guardar

# 3. Compara contra la configuración del material de referencia
python main.py --corpus tu_texto.txt --sin-negadores
```

Si el archivo no existe, el error lista los corpus disponibles en `data/raw/` en lugar de fallar con un `FileNotFoundError` sin contexto.

### Requisitos del texto

| Criterio | Razón |
|---|---|
| **UTF-8, sin doble codificación** | `carga.py` aborta si detecta secuencias `Ã`, `Â`, `â€` — firma de mojibake. Repara el archivo en origen, no en el pipeline. |
| **Español contemporáneo** | `es_core_news_sm` se entrenó sobre AnCora. Con español del Siglo de Oro el POS tagger degrada de forma medible. |
| **Prosa, no verso** | La segmentación en oraciones usa el parser sintáctico; el verso libre rompe `doc.sents`. |
| **Menos de 1,000,000 caracteres** | Límite de `nlp.max_length`. El pipeline falla con instrucción explícita si se supera; para textos mayores, procesar por capítulos con `nlp.pipe()`. |
| **Más de 100 KB, idealmente** | Con textos cortos la TTR se acerca a 1.0 y ninguna técnica de normalización puede comprimir de forma apreciable. |
| **Derechos de autor verificados** | Ver la sección siguiente. |

Cada corpus escribe sus resultados en su propio subdirectorio de `outputs/`, así que las corridas no se sobreescriben y el nombre del directorio documenta con qué dato se generó cada tabla.

### Interpretar tus resultados

- **TTR alta (> 0.7)** → corpus corto o muy variado; una reducción baja es esperada, no un fallo.
- **Filtrado > 70 %** → revisa si el texto es una adaptación pedagógica o un material con densidad anómala de partículas gramaticales.
- **Divergencia stem/lema** → inspecciona los grupos a mano antes de atribuir la culpa a una técnica.
- **`no_ascii` cercano a 0** en un texto en español → los acentos se perdieron en algún paso previo.

---

## Procedencia de los corpus

Documentar el origen de cada dato es parte del trabajo. Dos textos visualmente idénticos pueden tener propiedades distintas, y descubrirlo tarde invalida el análisis.

| Archivo | Obra | Nota |
|---|---|---|
| `cap1_principito.txt` | *El Principito*, cap. 1 — Antoine de Saint-Exupéry, traducción de Bonifacio del Carril | Incluido en el repositorio por requisito de la actividad evaluada. **No es de dominio público en México**: la ley federal establece vida del autor más 100 años, y Saint-Exupéry falleció en 1944. Se incluye un solo capítulo, con fines exclusivamente académicos y sin ánimo de lucro. |
| `caperucita_roja.txt` | *Caperucita Roja* — relato folclórico | El relato carece de autor con derechos vigentes, pero la traducción o adaptación concreta puede tener derechos propios. *Completar con la fuente exacta utilizada.* |

Durante el desarrollo, una versión previa del corpus resultó ser una **adaptación pedagógica para estudiantes de español como segunda lengua**, no la traducción canónica. Su densidad anómala de pronombres explícitos y muletillas habría distorsionado todas las métricas. Pasó todas las verificaciones técnicas —UTF-8 válido, sin mojibake, tamaño suficiente— y solo la inspección del contenido lo detectó.

**Criterio general:** el código se versiona; los datos se referencian. `data/raw/*.txt` está excluido por defecto en `.gitignore`, con una excepción explícita y documentada para el corpus del entregable. En producción, un corpus vive en almacenamiento de objetos con un puntero versionado (DVC, git-lfs), no dentro del repositorio.

---

## Tests

```bash
pytest -v
```

| Test | Qué verifica |
|---|---|
| `test_limpiar_lema_descarta_clitico` | Los pronombres enclíticos se eliminan del lema del verbo |
| `test_la_negacion_sobrevive_al_filtrado` | `"no falló"` y `"falló"` producen listas de lemas distintas |
| `test_reproduce_el_fallo_del_curso` | Con `PRESERVAR_NEGADORES=False`, ambas frases colapsan al mismo resultado |
| `test_ningun_lema_contiene_espacios` | Ningún lema queda como cadena multipalabra |
| `test_no_quedan_signos_de_puntuacion` | El filtrado elimina la puntuación |
| `test_el_lematizador_transforma_algo` | La lista de transformaciones no viene vacía (detecta modo `lookup`) |

`test_reproduce_el_fallo_del_curso` **codifica el defecto como comportamiento esperado** bajo una configuración explícita. Si algún día empieza a fallar, significa que la lista de stop words de spaCy cambió — y el aviso llega por el test, no por resultados anómalos varios módulos después.

---

## Decisiones de diseño

| Decisión | Razón |
|---|---|
| `data/raw/` inmutable | Ningún proceso escribe ahí. Un bug que sobreescriba el corpus original es irrecuperable. |
| Modelo fijado por URL del wheel | `spacy download` en runtime rompe builds sin red e instala versiones no fijadas. |
| Lematizar **antes** de bajar a minúsculas | El POS tagger usa la señal de mayúscula, y el lematizador de reglas depende del POS. El orden inverso degrada los lemas en silencio, sin excepción. |
| Doble verificación de stop words (forma y lema) | Una conjugación puede pasar el filtro de forma superficial y lematizarse a un lema que sí estaba en la lista. |
| `parser` activo | `doc.sents` lo requiere para la segmentación en oraciones. |
| `ner` desactivado | No se extraen entidades; ahorra cómputo sin pérdida. |
| `lru_cache` en la carga del modelo | Evita releer ~13 MB de disco en cada invocación. |
| Validación de encoding en la capa de carga | La doble codificación es UTF-8 *válido*: `file` no la detecta y ninguna excepción se dispara. |
| `raise` si formas y lemas difieren en longitud | Convierte un error de interpretación en un error imposible de ignorar. |
| `requirements.txt` + `requirements.lock.txt` | Intención y estado resuelto son cosas distintas. Fijar solo las dependencias directas deja las transitivas al azar. |

---

## Licencias

- **Código:** ver `LICENSE` (por definir).
- **Modelo `es_core_news_sm`:** **GPL-3.0**, no MIT como los modelos en inglés — deriva del corpus AnCora, con licencia copyleft. Irrelevante para uso académico; decisivo si el proyecto se distribuyera comercialmente.
- **Corpus:** ver la sección de procedencia.
