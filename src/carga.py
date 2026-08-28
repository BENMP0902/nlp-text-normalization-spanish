"""Lectura y saneamiento del texto crudo.

Esta capa existe porque `open(ruta, encoding='utf-8')` no deja el texto limpio.
Cada saneamiento aquí corresponde a un fallo real observado, no a paranoia.
"""
import unicodedata
from pathlib import Path

# Secuencias que solo aparecen si el texto pasó por una doble codificación.
# 'Ã' + cualquier cosa es la firma de bytes UTF-8 releídos como Latin-1.
_FIRMAS_MOJIBAKE = ("Ã", "Â", "â€", "ï¿½")


def detectar_mojibake(texto: str) -> list[str]:
    """Devuelve las firmas de doble codificación encontradas en el texto.

    Lista vacía = texto limpio. No repara: solo detecta. Reparar datos crudos
    en tiempo de carga oculta el problema; queremos que grite.
    """
    return [f for f in _FIRMAS_MOJIBAKE if f in texto]


def cargar_texto(ruta: Path) -> str:
    """Lee un archivo de texto y lo deja en un estado canónico."""
    if not ruta.exists():                       # falla ruidoso y temprano
        raise FileNotFoundError(
            f"No existe {ruta}. ¿Copiaste cap1_principito.txt a data/raw/?"
        )

    # 1) encoding='utf-8-sig': consume el BOM (Byte Order Mark) si lo hay.
    #    Notepad de Windows y Out-File de PowerShell lo escriben por defecto.
    #    Con 'utf-8' a secas, tu primer token sería '\ufeffCuando' en vez de
    #    'Cuando' — bug silencioso que solo ves inspeccionando bytes.
    texto = ruta.read_text(encoding="utf-8-sig")

    # 2) CRLF (Windows) → LF (POSIX). spaCy trata '\r' como token de espacio:
    #    no rompe, pero ensucia los conteos entre ejecuciones de distinto SO.
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")

    # 3) Guarda contra doble codificación. Va DESPUÉS de normalizar saltos y
    #    ANTES de NFC, porque NFC no repara mojibake — lo consolida.
    if firmas := detectar_mojibake(texto):
        raise ValueError(
            f"{ruta} tiene doble codificación (mojibake). Firmas: {firmas}. "
            "Repara el archivo en origen; NO lo parchees aquí."
        )

    # 4) Normalización Unicode a NFC (Unicode Standard Annex #15).
    #    'á' puede ser UN codepoint (U+00E1, NFC) o DOS (a + U+0301, NFD).
    #    Idénticos a la vista, distintos para el intérprete. macOS produce NFD;
    #    sin esto tu vocabulario tendría entradas duplicadas invisibles.
    texto = unicodedata.normalize("NFC", texto)

    # 5) Espacio duro (U+00A0) → espacio normal. Frecuente en texto copiado de
    #    web o PDF; el tokenizador lo trata como carácter, no como separador.
    texto = texto.replace("\u00a0", " ")

    if not texto.strip():
        raise ValueError(f"{ruta} está vacío o solo tiene espacios.")

    return texto


def diagnostico(texto: str) -> dict:
    """Métricas de verificación de la carga. Míralas, no las ignores."""
    return {
        "caracteres": len(texto),
        "lineas": texto.count("\n") + 1,
        # Acentos, ñ y comillas tipográficas viven arriba de ASCII. Si este
        # número es 0 en un texto en español, algo se comió los acentos.
        "no_ascii": sum(1 for c in texto if ord(c) > 127),
        "primeros_60": repr(texto[:60]),   # repr() delata BOM, \r y escapes
    }