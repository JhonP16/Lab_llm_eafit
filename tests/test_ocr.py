import pytest
from PIL import Image, ImageDraw, ImageFont

from src import ocr


def _image(text: str) -> Image.Image:
    img = Image.new("RGB", (900, 200), "white")
    ImageDraw.Draw(img).text((30, 60), text, fill="black", font=ImageFont.load_default(size=56))
    return img


@pytest.mark.skipif(not ocr.tesseract_languages(), reason="Tesseract no instalado")
def test_tesseract_reads_simple_text():
    res = ocr.extract_text(_image("HELLO WORLD"), lang="eng")
    assert "HELLO" in res.text.upper() and res.n_words >= 1
    assert res.confidence is None or res.confidence > 30


def test_preprocess_upscales_small_images():
    out = ocr.preprocess(Image.new("RGB", (200, 100), "white"), min_side=800)
    assert out.mode == "L" and max(out.size) == 800
