import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.segmentacion import segmentar_en_parrafos
from src.vectorizacion import construir_bow, construir_tfidf, matriz_similitud


@pytest.fixture
def corpus():
    """Corpus mínimo: 'zorro' está en TODOS los documentos (df = n)."""
    return [
        ["zorro", "rosa", "planeta", "rosa"],
        ["zorro", "lobo", "bosque", "lobo"],
        ["zorro", "rosa", "domesticar", "planeta"],
    ]


def test_idf_de_termino_ubicuo_es_uno_no_cero(corpus):
    """
    Documenta el error del material del curso: con smooth_idf=True (default),
    idf = ln((1+n)/(1+df)) + 1, así que df=n ⇒ idf = ln(1) + 1 = 1.0.
    NO se anula. Si sklearn cambiara esto, quiero enterarme por un test.
    """
    _, v = construir_tfidf(corpus, min_df=1, max_df=1.0)
    idf = dict(zip(v.get_feature_names_out(), v.idf_))
    assert idf["zorro"] == pytest.approx(1.0)


def test_max_df_si_elimina_el_termino_ubicuo(corpus):
    """max_df hace lo que el material creía que hacía el IDF."""
    _, v = construir_tfidf(corpus, min_df=1, max_df=0.99)
    assert "zorro" not in v.get_feature_names_out()


def test_no_hay_re_tokenizacion(corpus):
    """
    El default de sklearn (token_pattern=r'\\b\\w\\w+\\b') descartaría tokens de
    un carácter y partiría los guionados. Nuestro analyzer de identidad no.
    """
    corpus_raro = [["e", "no", "auto-crítica"], ["e", "sí", "auto-crítica"]]
    _, v = construir_bow(corpus_raro, min_df=1)
    vocab = set(v.get_feature_names_out())
    assert {"e", "auto-crítica"} <= vocab


def test_filas_tfidf_normalizadas_l2(corpus):
    """norm='l2' ⇒ cada documento vive en la esfera unitaria."""
    X, _ = construir_tfidf(corpus, min_df=1, max_df=1.0)
    normas = np.sqrt(X.multiply(X).sum(axis=1)).A.ravel()
    assert np.allclose(normas, 1.0)


def test_similitud_diagonal_unitaria(corpus):
    X, _ = construir_tfidf(corpus, min_df=1, max_df=1.0)
    assert np.allclose(np.diag(matriz_similitud(X)), 1.0)


def test_segmentacion_descarta_fragmentos_cortos():
    texto = "Cap. 1\n\n" + ("palabra " * 40) + "\n\ncorto"
    assert len(segmentar_en_parrafos(texto, min_caracteres=150)) == 1
