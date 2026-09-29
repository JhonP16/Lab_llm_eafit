from src import llm


def test_messages_contain_tone_and_delimited_ocr():
    msgs = llm.build_messages("texto ocr", "Técnica", "Breve", "Español", extra="sé conciso")
    assert "terminología precisa" in msgs[0]["content"]
    assert "<texto_ocr>\ntexto ocr\n</texto_ocr>" in msgs[1]["content"]
    assert "sé conciso" in msgs[1]["content"]
    formal = llm.build_messages("x", "Formal", "Media", "English")[0]["content"]
    assert "formal" in formal.lower() and "English" in formal


def test_reasoning_models_skip_sampling_params():
    assert llm.is_reasoning_model("o3-mini") and llm.is_reasoning_model("gpt-5")
    assert not llm.is_reasoning_model("gpt-4o-mini") and not llm.is_reasoning_model("gpt-5-chat-latest")
    k = llm.LLMParams(model="o3-mini", temperature=1.2).to_kwargs()
    assert "temperature" not in k and k["max_completion_tokens"] == 1200
    k = llm.LLMParams(model="gpt-4o-mini", temperature=0.2, seed=7).to_kwargs()
    assert k["temperature"] == 0.2 and k["seed"] == 7
