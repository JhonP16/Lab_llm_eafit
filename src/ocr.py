"""OCR: carga/preprocesado de imágenes y motores Tesseract (por defecto) y EasyOCR (opcional)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageFilter, ImageOps

TESSERACT_TO_EASYOCR = {"spa": "es", "eng": "en", "fra": "fr", "deu": "de", "por": "pt", "ita": "it"}

PSM_MODES = {
    3: "Automático (por defecto)",
    4: "Una sola columna de texto",
    6: "Bloque de texto uniforme",
    11: "Texto disperso (sin orden)",
}


class OCRError(RuntimeError):
    """Error controlado del OCR con mensaje apto para mostrar al usuario."""


@dataclass
class OCRResult:
    text: str
    engine: str
    confidence: float | None  # 0-100, promedio por palabra; None si el motor no la da
    n_words: int


def load_image(file: BinaryIO) -> Image.Image:
    """Abre la imagen respetando la orientación EXIF y la convierte a RGB."""
    img = Image.open(file)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def preprocess(img: Image.Image, min_side: int = 1400) -> Image.Image:
    """Escala de grises + autocontraste + ampliación de imágenes pequeñas + enfoque."""
    gray = ImageOps.autocontrast(img.convert("L"))
    w, h = gray.size
    if max(w, h) < min_side:
        k = min_side / max(w, h)
        gray = gray.resize((round(w * k), round(h * k)), Image.LANCZOS)
    return gray.filter(ImageFilter.SHARPEN)


# --------------------------------------------------------------------------- Tesseract
def tesseract_languages() -> list[str]:
    """Idiomas instalados en Tesseract ([] si Tesseract no está disponible)."""
    try:
        import pytesseract

        return sorted(l for l in pytesseract.get_languages(config="") if l != "osd")
    except Exception:
        return []


def _run_tesseract(img: Image.Image, lang: str, psm: int) -> OCRResult:
    try:
        import pytesseract
        from pytesseract import Output, TesseractNotFoundError
    except ImportError as e:  # pragma: no cover
        raise OCRError("Falta la librería 'pytesseract' (pip install pytesseract).") from e

    try:
        data = pytesseract.image_to_data(img, lang=lang, config=f"--psm {psm}", output_type=Output.DICT)
    except TesseractNotFoundError as e:
        raise OCRError(
            "No se encontró el binario de Tesseract. Instálalo (Ubuntu: `sudo apt install tesseract-ocr "
            "tesseract-ocr-spa`; macOS: `brew install tesseract tesseract-lang`; Windows: instalador UB-Mannheim)."
        ) from e
    except pytesseract.TesseractError as e:
        raise OCRError(f"Tesseract falló (¿está instalado el idioma '{lang}'?): {e}") from e

    lines: dict[tuple[int, int, int], list[str]] = {}
    confs: list[float] = []
    for i, raw in enumerate(data["text"]):
        word = str(raw).strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        conf = float(data["conf"][i])
        if conf >= 0:
            confs.append(conf)

    out: list[str] = []
    prev: tuple[int, int] | None = None
    for (block, par, _line), ws in lines.items():
        if prev is not None and (block, par) != prev:
            out.append("")  # línea en blanco entre párrafos
        out.append(" ".join(ws))
        prev = (block, par)

    text = "\n".join(out).strip()
    n_words = sum(len(v) for v in lines.values())
    return OCRResult(text, "Tesseract", float(np.mean(confs)) if confs else None, n_words)


# --------------------------------------------------------------------------- EasyOCR
@lru_cache(maxsize=2)
def _easyocr_reader(langs: tuple[str, ...]):
    import easyocr  # import diferido: es opcional y pesado

    return easyocr.Reader(list(langs), gpu=False, verbose=False)


def _run_easyocr(img: Image.Image, lang: str) -> OCRResult:
    try:
        reader = _easyocr_reader(tuple(TESSERACT_TO_EASYOCR.get(l, l) for l in lang.split("+")))
    except ImportError as e:
        raise OCRError("EasyOCR no está instalado. Usa: pip install -r requirements-extras.txt") from e

    results = reader.readtext(np.array(img.convert("RGB")), detail=1, paragraph=False)
    texts = [str(t).strip() for _bbox, t, _conf in results if str(t).strip()]
    confs = [float(c) * 100 for _bbox, t, c in results if str(t).strip()]
    return OCRResult(
        "\n".join(texts).strip(),
        "EasyOCR",
        float(np.mean(confs)) if confs else None,
        sum(len(t.split()) for t in texts),
    )


# --------------------------------------------------------------------------- API pública
def extract_text(
    img: Image.Image,
    engine: str = "Tesseract",
    lang: str = "spa+eng",
    psm: int = 3,
    enhance: bool = True,
) -> OCRResult:
    """Extrae texto de una imagen PIL. `lang` usa códigos de Tesseract unidos por '+'."""
    work = preprocess(img) if enhance else img
    if engine == "EasyOCR":
        return _run_easyocr(work, lang)
    return _run_tesseract(work, lang, psm)
