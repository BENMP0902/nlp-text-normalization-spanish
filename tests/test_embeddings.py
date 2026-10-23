"""Tests de la etapa 3: embeddings Word2Vec."""
import numpy as np
import pytest

from src import config
from src.carga import cargar_texto
from src.embeddings import diagnosticar_suficiencia, entrenar, preparar_corpus
from src.normalizacion import construir_corpus, procesar, procesar_por_partes
from src.segmentacion import segmentar_en_parrafos
from src.visualizacion import proyectar_pca


@pytest.fixture
def oraciones_minimas():
    """Corpus diminuto con cada palabra repetida por encima de min_count.

    Repetir es necesario: con min_count=5, un corpus sin repeticiones deja el
    vocabulario vacío y gensim lanza RuntimeError antes de entrenar.
    """
    base = [["el", "gato", "come", "pescado"], ["el", "perro", "come", "carne"],
            ["la", "niña", "mira", "el", "gato"], ["la", "niña", "mira", "el", "perro"]]
    return base * config.W2V_MIN_COUNT


def test_entrenamiento_conserva_stop_words():
    """Con stop_words=frozenset() el corpus de Word2Vec conserva las palabras función.

    Es la diferencia deliberada frente a las etapas 1 y 2. Si alguien quita
    el parámetro, la ventana de contexto pierde las palabras que la articulan
    y el test lo detecta.
    """
    oraciones = construir_corpus(procesar("El gato no come de la mesa."),
                                 stop_words=frozenset())
    lemas = oraciones[0]
    for palabra in ("el", "de", "no"):
        assert palabra in lemas
    assert "." not in lemas                     # la puntuación sí se descarta


def test_construir_corpus_sin_argumento_conserva_comportamiento_previo():
    """El default sigue filtrando stop words: las etapas 1 y 2 no cambian."""
    lemas = construir_corpus(procesar("El gato come de la mesa."))[0]
    assert "el" not in lemas and "de" not in lemas


def test_troceo_por_parrafo_no_parte_oraciones():
    """Cada parte termina en un final de oración real, nunca a la mitad.

    Una línea dura a mitad de oración (texto envuelto a 80 columnas) NO es
    frontera: segmentar_en_parrafos solo corta en líneas en blanco.
    """
    texto = ("Se puso el sol. El viajero siguió\nadelante por la vereda.\n\n"
             "Era un hombre de mediana edad.")
    partes = segmentar_en_parrafos(texto, min_caracteres=1)
    docs = list(procesar_por_partes(partes))

    assert len(docs) == 2
    for doc in docs:
        assert doc[-1].text == "."
    # La oración partida por el salto de línea sigue siendo una sola.
    assert len(list(docs[0].sents)) == 2


def test_raya_ascii_se_separa_de_la_palabra(tmp_path):
    """'--' pegado a la palabra se convierte en raya separada.

    Sin esto, 'Nela--' y '--Nela' entran al vocabulario como palabras
    distintas de 'Nela' y fragmentan sus contextos.
    """
    ruta = tmp_path / "dialogo.txt"
    ruta.write_text("--¿Y tú, Nela?--dijo el doctor.\n", encoding="utf-8")
    oraciones, _ = preparar_corpus(cargar_texto(ruta))
    lemas = [w for o in oraciones for w in o]
    assert "nela" in lemas
    assert not any("-" in w or "—" in w for w in lemas)


def test_misma_semilla_da_vectores_identicos(oraciones_minimas):
    """Con workers=1 y la misma semilla los vectores coinciden bit a bit.

    Dentro de un mismo proceso el hash de str es estable, así que esta prueba
    no depende de PYTHONHASHSEED; entre procesos sí depende (ver README).
    """
    a = entrenar(oraciones_minimas, semilla=7)
    b = entrenar(oraciones_minimas, semilla=7)
    c = entrenar(oraciones_minimas, semilla=8)
    assert np.array_equal(a.wv.vectors, b.wv.vectors)
    assert not np.array_equal(a.wv.vectors, c.wv.vectors)   # la semilla importa


def test_corpus_minimo_dispara_aviso(oraciones_minimas):
    """Sobre un corpus diminuto el diagnóstico advierte en vez de callar."""
    modelo = entrenar(oraciones_minimas)
    diag = diagnosticar_suficiencia(modelo, oraciones_minimas, ["gato", "perro"])
    assert any("Corpus insuficiente" in a for a in diag["avisos"])


def test_pca_centra_la_matriz():
    """La proyección PCA tiene media cero aunque los vectores compartan un offset.

    Es lo que la distingue de TruncatedSVD: sin centrar, un offset común a
    todas las filas se llevaría la primera componente.
    """
    rng = np.random.default_rng(0)
    M = rng.normal(size=(30, 10)) + 5.0         # offset común grande
    coords, var = proyectar_pca(M)
    assert np.allclose(coords.mean(axis=0), 0.0, atol=1e-10)
    assert 0 < var.sum() <= 1
