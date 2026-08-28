"""Tests del pipeline. El primero justifica la existencia del proyecto."""
import pytest

from src import config
from src.normalizacion import procesar, lematizar, limpiar_lema


@pytest.fixture
def preservar_negadores():
    """Fija PRESERVAR_NEGADORES=True y restaura el valor previo al terminar.

    Fixture con yield: lo de arriba corre antes del test, lo de abajo después,
    incluso si el test falla. Sustituye a mutar config a mano, patrón que crea
    dependencia del ORDEN de ejecución — y pytest no garantiza orden.
    """
    previo = config.PRESERVAR_NEGADORES
    config.PRESERVAR_NEGADORES = True
    yield
    config.PRESERVAR_NEGADORES = previo


@pytest.fixture
def filtrar_negadores():
    """Reproduce el comportamiento del curso: los negadores SÍ se filtran."""
    previo = config.PRESERVAR_NEGADORES
    config.PRESERVAR_NEGADORES = False
    yield
    config.PRESERVAR_NEGADORES = previo


def test_limpiar_lema_descarta_clitico():
    """Test unitario puro: no carga el modelo, corre en microsegundos.

    Separar la lógica pura de la que depende de spaCy es lo que permite tener
    una suite rápida. Si todos tus tests cargan un modelo de 13 MB, dejas de
    ejecutarlos, y una suite que no se ejecuta no protege nada.
    """
    assert limpiar_lema("masticar él") == "masticar"
    assert limpiar_lema("tener") == "tener"          # idempotente sin clítico
    assert limpiar_lema("dar él él") == "dar"        # doble clítico: dárselo


def test_la_negacion_sobrevive_al_filtrado(preservar_negadores):
    """El test que el material del curso no pasaría.

    Si el pipeline elimina 'no', ambas frases producen el MISMO vector con
    polaridad OPUESTA. Convierte una afirmación teórica en verificación
    ejecutable: la diferencia entre 'creo que está bien' y 'está probado'.
    """
    lemas_neg, _ = lematizar(procesar("El producto no falló."))
    lemas_pos, _ = lematizar(procesar("El producto falló."))

    assert lemas_neg != lemas_pos, "El filtrado destruyó la negación"
    assert "no" in lemas_neg


def test_reproduce_el_fallo_del_curso(filtrar_negadores):
    """Documenta el comportamiento defectuoso como aserción, no como opinión."""
    lemas_neg, _ = lematizar(procesar("El producto no falló."))
    lemas_pos, _ = lematizar(procesar("El producto falló."))

    assert lemas_neg == lemas_pos    # el bug, hecho test


def test_ningun_lema_contiene_espacios(preservar_negadores):
    """Guarda contra los pronombres enclíticos del español.

    Frase tomada del corpus real. Sin limpiar_lema(), 'masticarla' produce
    el lema 'masticar él' — dos palabras en un término de vocabulario, que
    el vectorizador del módulo 4 volvería a partir en dos.
    """
    lemas, _ = lematizar(procesar("La boa se traga su presa sin masticarla."))

    assert all(" " not in l for l in lemas), f"Lema con espacio: {lemas}"
    assert "masticar" in lemas


def test_no_quedan_signos_de_puntuacion(preservar_negadores):
    lemas, _ = lematizar(procesar("Dibujé, entonces, mi primer dibujo."))
    assert not any(l in {",", ".", ";", ":"} for l in lemas)


def test_el_lematizador_transforma_algo(preservar_negadores):
    """Si transformaciones viene vacía, el lematizador no está analizando."""
    _, transformaciones = lematizar(
        procesar("Las personas mayores necesitaban explicaciones detalladas.")
    )
    assert transformaciones, "Cero transformaciones: ¿lematizador en modo lookup?"
