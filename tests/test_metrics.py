import zlib
from types import SimpleNamespace

import numpy as np

from src import metrics

ES = ("La fotosíntesis es el proceso mediante el cual las plantas convierten la luz en energía química. "
      "Además, este proceso libera oxígeno a la atmósfera, porque el agua se descompone en la reacción. "
      "Por lo tanto, sin fotosíntesis la vida tal como la conocemos no sería posible.")
EN = "The system stores data in memory. However, it also writes a copy to disk, which improves durability."


class FakeEmbeddings:
    """Cliente falso: embeddings de bolsa de palabras hasheadas (determinista y sin red)."""

    def __init__(self):
        self.embeddings = self

    def create(self, model, input):
        data = []
        for i, text in enumerate(input):
            v = np.zeros(64)
            for w in metrics.words(text):
                v[zlib.crc32(w.lower().encode()) % 64] += 1
            data.append(SimpleNamespace(index=i, embedding=v.tolist() if v.any() else [1.0] + [0.0] * 63))
        return SimpleNamespace(data=data)


def test_split_sentences_and_words():
    assert len(metrics.split_sentences(ES)) == 3
    assert metrics.words("**Hola** mundo, ¿qué tal?") == ["Hola", "mundo", "qué", "tal"]


def test_ocr_sentences_ignore_hard_line_wraps():
    text = "Esta oración se corta\nen dos líneas. Otra oración."
    assert len(metrics.split_sentences(text, newline_breaks=False)) == 2
    assert len(metrics.split_sentences(text, newline_breaks=True)) == 3


def test_detect_lang():
    assert metrics.detect_lang(ES) == "es"
    assert metrics.detect_lang(EN) == "en"


def test_syllables_spanish():
    assert metrics._syllables_es("cuando") == 2
    assert metrics._syllables_es("día") == 2
    assert metrics._syllables_es("que") == 1
    assert metrics._syllables_es("fotosíntesis") == 5


def test_basic_and_readability():
    b = metrics.basic_stats(ES)
    assert b["sentences"] == 3 and b["words"] > 30
    assert 0 < b["lexical_diversity_mattr"] <= 1
    r = metrics.readability(ES, "es")
    assert r["index"].startswith("Szigriszt") and isinstance(r["score"], float)
    assert metrics.readability(EN, "en")["index"] == "Flesch Reading Ease"


def test_syntax_counts_connectors():
    s = metrics.syntax_metrics(ES, "es")
    assert s["connectors_per_100_words"] > 0
    assert s["subordinators_per_sentence"] > 0
    assert s["sentence_len_max"] >= s["sentence_len_mean"]


def test_semantic_and_coherence_with_fake_client():
    ocr = "Las plantas convierten la luz en energía química."
    out = metrics.semantic_and_coherence(FakeEmbeddings(), ocr, ES, "fake")
    assert 0 < out["semantic"]["similarity_ocr_vs_response"] <= 1
    assert len(out["coherence"]["adjacent_similarity"]) == 2
    assert out["coherence"]["weakest_transition_at"] in (1, 2)


def test_compute_all_without_client_and_error_isolation():
    m = metrics.compute_all("texto", ES, client=None)
    assert m["lang"] == "es" and "semantic" not in m and not m["errors"]

    class Broken:
        embeddings = SimpleNamespace(create=lambda **k: (_ for _ in ()).throw(RuntimeError("sin red")))

    m = metrics.compute_all("texto de prueba.", ES, client=Broken(), use_judge=False)
    assert "embeddings" in m["errors"] and m["basic"]["words"] > 0


def test_judge_json_parsing_and_clamping():
    raw = '```json\n{"coherencia": 130, "cohesion": "80", "semantica": null, "sintaxis": 70, "gramatica": 90}\n```'
    data = metrics._parse_json(raw)
    assert data["coherencia"] == 130
