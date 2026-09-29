from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_renders_without_upload():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("Sube una imagen" in i.value for i in at.info)
