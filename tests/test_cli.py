"""Pruebas de humo (smoke tests): verifican que el sistema ENCIENDE.

Existen porque la suite anterior tenía 12 tests en verde mientras el CLI no
arrancaba: ningún test importaba main.py ni visualizacion.py, así que un
NameError en el punto de entrada y una dependencia faltante pasaban
inadvertidos. Estas pruebas no miden calidad; miden que haya algo que medir.
"""
import re
import subprocess
import sys
import importlib
import importlib.metadata as metadata
from pathlib import Path

import pytest

# parents[1] sube de tests/ a la raíz del proyecto. Calcularlo desde __file__
# y no desde os.getcwd() hace que la suite funcione sin importar desde qué
# directorio se invoque pytest.
RAIZ = Path(__file__).resolve().parents[1]


def _modulos_de_src() -> list[str]:
    """Descubre los módulos de src/ leyendo el directorio.

    Dinámico y no una lista fija: un módulo nuevo (embeddings.py, por
    ejemplo) queda cubierto en cuanto se crea. Una lista escrita a mano
    protege solo lo que alguien se acordó de agregar, que es justo lo que
    falla cuando hay prisa.

    sorted() fija el orden: así los IDs de los tests parametrizados no
    cambian entre corridas y un fallo siempre se llama igual en el log.
    """
    return sorted(
        p.stem for p in (RAIZ / "src").glob("*.py")
        if not p.stem.startswith("_")        # excluye __init__.py
    )


def _normalizar(nombre: str) -> str:
    """Normaliza un nombre de paquete según PEP 503.

    Minúsculas, y cualquier racha de '-', '_' o '.' colapsa a un solo guion.
    Hace falta porque 'scikit_learn', 'scikit-learn' y 'Scikit-Learn' son el
    mismo paquete para pip, pero tres cadenas distintas para Python.
    """
    return re.sub(r"[-_.]+", "-", nombre).lower()


def _paquetes_declarados(ruta: Path) -> list[str]:
    """Extrae los nombres de paquete de un requirements.txt.

    Maneja las cuatro formas que aparecen en este proyecto:
      paquete==1.2.3
      paquete>=1.0
      paquete[extra]==1.0
      paquete @ https://.../archivo.whl#sha256=...   (PEP 508, referencia directa)

    No usa `packaging.requirements` para no agregar una dependencia solo
    para la suite de pruebas. Si el archivo creciera en complejidad
    (marcadores de entorno, restricciones), convendría cambiar de opinión.
    """
    nombres = []
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.split("#", 1)[0].strip()        # descarta comentarios
        if not linea or linea.startswith("-"):         # vacías y flags (-r, -e)
            continue
        if " @ " in linea:                             # referencia directa
            nombres.append(linea.split(" @ ", 1)[0].strip())
            continue
        # Corta en el primer separador de versión, extra o marcador
        nombre = re.split(r"[<>=!~\[;]", linea, maxsplit=1)[0].strip()
        if nombre:
            nombres.append(nombre)
    return nombres


# ---------------------------------------------------------------------------
# ¿Arranca?
# ---------------------------------------------------------------------------

def test_el_cli_arranca():
    """Ejecuta `python main.py --help` en un subproceso.

    En subproceso y no con import: así se ejercita el intérprete real con su
    PATH y sus dependencias. Un import dentro de pytest no reproduce las
    condiciones de la invocación real.

    sys.executable es el intérprete que está corriendo pytest — el del venv
    si pytest se invocó con el venv activo. Usarlo, y no la cadena "python",
    evita que el subproceso caiga en el Python global por el PATH.

    `--help` hace que argparse imprima y salga con código 0 sin ejecutar el
    pipeline: la prueba es rápida y no depende de que exista un corpus.
    """
    r = subprocess.run([sys.executable, "main.py", "--help"],
                       cwd=RAIZ, capture_output=True, text=True)
    assert r.returncode == 0, f"El CLI no arranca:\n{r.stderr}"


@pytest.mark.parametrize("modulo", _modulos_de_src())
def test_el_modulo_importa(modulo):
    """Importa cada módulo de src/, uno por test.

    Parametrizado y no un bucle dentro de un solo test: con parametrize, el
    ID del test lleva el nombre del módulo (test_el_modulo_importa[carga]),
    así que el log dice CUÁL falló sin leer el traceback. Un bucle además se
    detiene en el primer fallo y oculta los demás.
    """
    importlib.import_module(f"src.{modulo}")


# ---------------------------------------------------------------------------
# ¿El entorno es el declarado?  (configuration drift)
# ---------------------------------------------------------------------------

def test_todo_requirements_esta_instalado():
    """Verifica que cada paquete de requirements.txt esté instalado.

    Esta es la prueba que detecta la deriva real. `pip check` NO sirve para
    esto: solo valida que las dependencias DECLARADAS por los paquetes
    instalados estén satisfechas. matplotlib es una dependencia de hoja —
    solo la necesita nuestro código, y los paquetes que la mencionan
    (pandas, scipy) lo hacen bajo un `extra` que no instalamos. Con
    matplotlib ausente, `pip check` reporta "No broken requirements found".

    metadata.distribution() consulta los metadatos instalados, no importa el
    paquete: es rápido y funciona con nombres de distribución (scikit-learn)
    que no coinciden con el nombre de import (sklearn).
    """
    requirements = RAIZ / "requirements.txt"
    assert requirements.exists(), f"No existe {requirements}"

    faltantes = []
    for nombre in _paquetes_declarados(requirements):
        try:
            metadata.distribution(_normalizar(nombre))
        except metadata.PackageNotFoundError:
            faltantes.append(nombre)

    assert not faltantes, (
        f"Paquetes declarados en requirements.txt pero ausentes del entorno: "
        f"{faltantes}\nIntérprete: {sys.executable}\n"
        f"¿Instalaste con el venv activo? `source .venv/bin/activate && "
        f"pip install -r requirements.txt`"
    )


def test_no_hay_dependencias_rotas():
    """`pip check`: las dependencias instaladas son consistentes entre sí.

    Complementa al test anterior, no lo reemplaza. Aquel detecta ausencias
    respecto a lo declarado; este detecta conflictos de versión entre lo
    instalado (paquete A exige B>=2 y hay B==1.4).
    """
    r = subprocess.run([sys.executable, "-m", "pip", "check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"Entorno inconsistente:\n{r.stdout}{r.stderr}"
