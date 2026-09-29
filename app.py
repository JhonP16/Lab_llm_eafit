"""Imagen → OCR → LLM (respuesta ampliada formal/técnica) → métricas del texto generado."""
from __future__ import annotations

import json

import openai
import pandas as pd
import streamlit as st

from src import llm, metrics, ocr

st.set_page_config(page_title="OCR + LLM", page_icon="📝", layout="wide")

LABELS = {
    "characters": "Caracteres",
    "words": "Palabras",
    "sentences": "Oraciones",
    "paragraphs": "Párrafos",
    "unique_words": "Palabras únicas",
    "avg_word_length": "Longitud media de palabra (letras)",
    "words_per_sentence": "Palabras por oración",
    "lexical_diversity_mattr": "Diversidad léxica (MATTR, 0-1)",
    "trigram_repetition": "Repetición de trigramas (0-1, menor = mejor)",
    "sentence_len_mean": "Longitud media de oración (palabras)",
    "sentence_len_std": "Desviación de longitud de oración",
    "sentence_len_max": "Oración más larga (palabras)",
    "long_sentence_ratio": "Proporción de oraciones > 30 palabras",
    "commas_per_sentence": "Comas por oración",
    "subordinators_per_sentence": "Subordinantes por oración",
    "connectors_per_100_words": "Conectores discursivos por 100 palabras",
    "dep_depth_mean": "Profundidad media del árbol sintáctico",
    "dep_depth_max": "Profundidad máxima del árbol sintáctico",
    "verb_ratio": "Proporción de verbos",
    "noun_ratio": "Proporción de sustantivos",
}

for key, default in {"ocr_text": "", "ocr_meta": None, "result": None, "metrics": None,
                     "file_sig": None, "available_models": None}.items():
    st.session_state.setdefault(key, default)


@st.cache_data(show_spinner=False)
def installed_tesseract_langs() -> list[str]:
    return ocr.tesseract_languages()


def explain_error(e: Exception) -> str:
    if isinstance(e, openai.AuthenticationError):
        return "API key inválida o sin permisos."
    if isinstance(e, openai.RateLimitError):
        return "Límite de uso o cuota agotada en tu cuenta de OpenAI."
    if isinstance(e, openai.NotFoundError):
        return "El modelo no existe o tu cuenta no tiene acceso a él."
    if isinstance(e, openai.APIConnectionError):
        return "No se pudo conectar con la API. Revisa tu conexión o la Base URL."
    return f"{type(e).__name__}: {e}"


def fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


# =========================================================================== barra lateral
with st.sidebar:
    st.header("Configuración")

    with st.expander("OpenAI", expanded=True):
        api_key = st.text_input("API key", type="password", placeholder="sk-...",
                                help="Se mantiene solo en la memoria de la sesión; no se guarda en disco.")
        base_url = st.text_input("Base URL (opcional)", placeholder="https://api.openai.com/v1")
        if st.button("Verificar key y cargar modelos", width="stretch"):
            if not api_key:
                st.warning("Ingresa tu API key primero.")
            else:
                try:
                    with st.spinner("Consultando modelos…"):
                        found = llm.list_chat_models(llm.make_client(api_key, base_url))
                    st.session_state.available_models = found or None
                    st.success(f"Key válida · {len(found)} modelos de chat disponibles.")
                except openai.OpenAIError as e:
                    st.error(explain_error(e))

        options = st.session_state.available_models or llm.DEFAULT_MODELS
        choice = st.selectbox("Modelo", [*options, "Otro…"],
                              index=options.index("gpt-4o-mini") if "gpt-4o-mini" in options else 0)
        model = st.text_input("Nombre del modelo", value="gpt-4o-mini") if choice == "Otro…" else choice

    reasoning = llm.is_reasoning_model(model)
    with st.expander("Parámetros del LLM", expanded=True):
        if reasoning:
            st.caption("Modelo de razonamiento: no admite temperature/top_p/penalizaciones. "
                       "Sube «Máx. tokens» (≥ 4000): el razonamiento también consume tokens de salida.")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05, disabled=reasoning,
                                help="Mayor = más variado/creativo; menor = más determinista.")
        top_p = st.slider("Top-p", 0.0, 1.0, 1.0, 0.05, disabled=reasoning)
        max_tokens = st.slider("Máx. tokens de salida", 100, 8000, 1200, 50)
        presence = st.slider("Presence penalty", -2.0, 2.0, 0.0, 0.1, disabled=reasoning)
        frequency = st.slider("Frequency penalty", -2.0, 2.0, 0.0, 0.1, disabled=reasoning)
        seed = st.number_input("Seed (0 = aleatorio)", min_value=0, value=0, step=1, disabled=reasoning)
        stream = st.checkbox("Mostrar la respuesta en streaming", value=True)

    with st.expander("OCR"):
        engine = st.radio("Motor", ["Tesseract", "EasyOCR"], horizontal=True,
                          help="EasyOCR es pesado y no se recomienda en Streamlit Cloud (límite de memoria). Usa Tesseract.")
        installed = installed_tesseract_langs()
        lang_opts = installed or ["spa", "eng"]
        langs = st.multiselect("Idiomas", lang_opts,
                               default=[l for l in ("spa", "eng") if l in lang_opts] or lang_opts[:1])
        ocr_lang = "+".join(langs) or "eng"
        psm = st.selectbox("Segmentación (Tesseract)", list(ocr.PSM_MODES),
                           format_func=lambda k: f"{k} · {ocr.PSM_MODES[k]}")
        enhance = st.checkbox("Mejorar imagen (grises, contraste, ampliar)", value=True)
        if engine == "Tesseract" and not installed:
            st.caption("No se detecta Tesseract en el sistema; se mostrará un error al ejecutar el OCR.")

    with st.expander("Métricas"):
        use_embeddings = st.checkbox("Semántica y coherencia (embeddings)", value=True)
        embedding_model = st.text_input("Modelo de embeddings", value="text-embedding-3-small")
        use_judge = st.checkbox("Evaluación por LLM (gramática, sintaxis…)", value=True)
        judge_model = st.text_input("Modelo evaluador", value="gpt-4o-mini")
        use_lt = st.checkbox("LanguageTool (gramática objetiva, requiere Java)", value=False)


# =========================================================================== render de métricas
def render_metrics(m: dict) -> None:
    st.subheader("Métricas del texto generado")
    for name, err in m["errors"].items():
        st.warning(f"Bloque «{name}» no disponible: {err}")

    sem, coh, judge, lt = m.get("semantic"), m.get("coherence") or {}, m.get("judge"), m.get("languagetool")
    tabs = st.tabs(["Resumen", "Estadísticas", "Sintaxis", "Semántica y coherencia", "Gramática"])

    with tabs[0]:
        b, r = m["basic"], m["readability"]
        cols = st.columns(5)
        cols[0].metric("Palabras", b["words"], help=f"Expansión ×{m['expansion_ratio']:.1f} respecto al texto OCR")
        cols[1].metric("Legibilidad", r.get("score", "–"), help=r.get("index"))
        cols[1].caption(r.get("level", ""))
        cols[2].metric("Similitud con OCR", f"{sem['similarity_ocr_vs_response']:.2f}" if sem else "–",
                       help="Coseno entre embeddings del texto OCR y de la respuesta.")
        cols[3].metric("Coherencia local", f"{coh['local_coherence']:.2f}" if "local_coherence" in coh else "–",
                       help="Similitud media entre oraciones consecutivas.")
        cols[4].metric("Gramática (LLM)", f"{judge['gramatica']}/100" if judge else "–")
        if judge:
            st.caption("Puntajes del evaluador LLM (0-100). Son una valoración subjetiva: úsalos de forma comparativa.")
            st.bar_chart(pd.DataFrame({"Puntaje": {k.capitalize(): judge[k] for k in metrics.JUDGE_SCORES}}),
                         height=260)
            if judge["comentario"]:
                st.info(judge["comentario"])

    with tabs[1]:
        rows = [(LABELS[k], fmt(v)) for k, v in m["basic"].items()]
        rd = m["readability"]
        if rd:
            rows += [(f"Legibilidad · {rd['index']}", f"{rd['score']} ({rd['level']})"),
                     ("Sílabas por palabra", fmt(rd["syllables_per_word"]))]
        st.dataframe(pd.DataFrame(rows, columns=["Métrica", "Valor"]), hide_index=True, width="stretch")

    with tabs[2]:
        syn = m["syntax"]
        st.caption(f"Analizador: {syn.get('parser', '–')}. Instala spaCy y su modelo para profundidad sintáctica y POS.")
        rows = [(LABELS[k], fmt(v)) for k, v in syn.items() if k in LABELS]
        st.dataframe(pd.DataFrame(rows, columns=["Métrica", "Valor"]), hide_index=True, width="stretch")

    with tabs[3]:
        if not sem:
            st.info("Activa «Semántica y coherencia (embeddings)» y proporciona la API key para ver estas métricas.")
        else:
            c = st.columns(3)
            c[0].metric("Similitud OCR ↔ respuesta", f"{sem['similarity_ocr_vs_response']:.3f}")
            c[1].metric("Cobertura del OCR", f"{sem['ocr_coverage']:.3f}",
                        help="Promedio de la mejor coincidencia de cada oración del OCR dentro de la respuesta.")
            c[2].metric("Contenido añadido", f"{sem['added_content']:.3f}",
                        help="1 − similitud media de cada oración generada con el OCR. Alto = aporta más allá del texto.")
            if "local_coherence" in coh:
                c = st.columns(3)
                c[0].metric("Coherencia local", f"{coh['local_coherence']:.3f}")
                c[1].metric("Coherencia global", f"{coh['global_coherence']:.3f}",
                            help="Similitud media de cada oración con el centroide del texto.")
                c[2].metric("Transición más débil", f"{coh['weakest_transition']:.3f}",
                            help=f"Entre las oraciones {coh['weakest_transition_at']} y {coh['weakest_transition_at'] + 1}.")
                adj = coh["adjacent_similarity"]
                st.line_chart(pd.DataFrame({"Similitud entre oraciones consecutivas": adj},
                                           index=range(1, len(adj) + 1)), height=220)
            st.caption("Los valores dependen del modelo de embeddings: compáralos entre respuestas del mismo modelo, "
                       "no contra un umbral absoluto.")

    with tabs[4]:
        if not judge and not lt:
            st.info("Activa la evaluación por LLM o LanguageTool en la barra lateral.")
        if judge:
            c = st.columns(3)
            c[0].metric("Gramática", f"{judge['gramatica']}/100")
            c[1].metric("Sintaxis", f"{judge['sintaxis']}/100")
            c[2].metric("Cohesión", f"{judge['cohesion']}/100")
            if judge["errores_gramaticales"]:
                st.markdown("**Errores señalados por el evaluador LLM**")
                st.dataframe(pd.DataFrame(judge["errores_gramaticales"]), hide_index=True, width="stretch")
            if judge["afirmaciones_dudosas"]:
                st.markdown("**Afirmaciones a verificar** (no aparecen en el texto fuente)")
                for a in judge["afirmaciones_dudosas"]:
                    st.markdown(f"- {a}")
        if lt:
            st.markdown("**LanguageTool**")
            c = st.columns(2)
            c[0].metric("Incidencias", lt["issue_count"])
            c[1].metric("Por cada 100 palabras", f"{lt['issues_per_100_words']:.2f}")
            if lt["issues"]:
                st.dataframe(pd.DataFrame(lt["issues"]), hide_index=True, width="stretch")


# =========================================================================== generación
def generate_response(client, ocr_text: str, tone: str, length: str, language: str, extra: str) -> bool:
    params = llm.LLMParams(model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                           presence_penalty=presence, frequency_penalty=frequency, seed=int(seed) or None)
    messages = llm.build_messages(ocr_text, tone, length, language, extra)
    usage: dict = {}
    try:
        with st.container(border=True):
            if stream:
                gen_text = st.write_stream(llm.stream_text(client, params, messages, usage))
            else:
                with st.spinner("Generando respuesta…"):
                    gen_text, usage = llm.complete_text(client, params, messages)
                st.markdown(gen_text)
    except openai.OpenAIError as e:
        st.error(explain_error(e))
        return False

    if not gen_text.strip():
        hint = " Sube «Máx. tokens»: el razonamiento consumió el presupuesto." if reasoning else ""
        st.warning("El modelo devolvió una respuesta vacía." + hint)
        return False

    lang = {"Español": "es", "English": "en"}.get(language)  # None = detección automática
    with st.spinner("Calculando métricas…"):
        m = metrics.compute_all(ocr_text, gen_text, lang=lang, client=client, use_embeddings=use_embeddings,
                                embedding_model=embedding_model, use_judge=use_judge, judge_model=judge_model,
                                use_languagetool=use_lt)
    st.session_state.result = {
        "text": gen_text,
        "usage": usage,
        "settings": {"model": model, "tone": tone, "length": length, "language": language, "extra": extra,
                     **{k: v for k, v in params.to_kwargs().items() if k not in ("model",)}},
    }
    st.session_state.metrics = m
    return True


# =========================================================================== página principal
st.title("OCR + ampliación con LLM")
st.caption("Sube una imagen, extrae su texto con OCR y obtén una versión ampliada (formal o técnica) con métricas de calidad. "
           "El texto extraído se envía a la API de OpenAI.")

uploaded = st.file_uploader("Imagen", type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"])
if uploaded is None:
    st.info("Sube una imagen para comenzar.")
    st.stop()

signature = (uploaded.name, uploaded.size)
if st.session_state.file_sig != signature:  # imagen nueva → limpiar estado previo
    st.session_state.update(file_sig=signature, ocr_text="", ocr_meta=None, result=None, metrics=None)

try:
    image = ocr.load_image(uploaded)
except Exception as e:  # noqa: BLE001
    st.error(f"No se pudo abrir la imagen: {e}")
    st.stop()

left, right = st.columns(2, gap="large")
with left:
    st.subheader("1 · Imagen")
    st.image(image, width="stretch")
    if st.button("Extraer texto (OCR)", type="primary", width="stretch"):
        try:
            with st.spinner("Ejecutando OCR…"):
                res = ocr.extract_text(image, engine=engine, lang=ocr_lang, psm=psm, enhance=enhance)
            st.session_state.ocr_text = res.text
            st.session_state.ocr_meta = {"engine": res.engine, "words": res.n_words, "confidence": res.confidence}
            st.session_state.result = st.session_state.metrics = None
            if not res.text:
                st.warning("No se detectó texto. Prueba otra segmentación, otro idioma o una imagen más nítida.")
        except ocr.OCRError as e:
            st.error(str(e))

with right:
    st.subheader("2 · Texto extraído")
    meta = st.session_state.ocr_meta
    if meta:
        c = st.columns(3)
        c[0].metric("Motor", meta["engine"])
        c[1].metric("Palabras", meta["words"])
        c[2].metric("Confianza media", f"{meta['confidence']:.0f}%" if meta["confidence"] is not None else "–")
    st.text_area("Texto (editable antes de enviarlo al LLM)", key="ocr_text", height=300,
                 placeholder="Pulsa «Extraer texto (OCR)» o escribe/pega el texto aquí.")

st.divider()
st.subheader("3 · Ampliación con LLM")
c1, c2, c3 = st.columns(3)
tone = c1.radio("Estilo de respuesta", list(llm.TONES), horizontal=True)
length = c2.select_slider("Extensión", list(llm.LENGTHS), value="Media")
language = c3.selectbox("Idioma de la respuesta", list(llm.LANGUAGES))
extra = st.text_input("Instrucción adicional (opcional)", placeholder="Ej.: enfócate en las implicaciones prácticas")

if st.button("Generar respuesta ampliada", type="primary"):
    if not api_key:
        st.error("Ingresa tu API key en la barra lateral.")
    elif not st.session_state.ocr_text.strip():
        st.error("Primero extrae o escribe el texto de partida.")
    else:
        ok = generate_response(llm.make_client(api_key, base_url), st.session_state.ocr_text, tone, length,
                               language, extra)
        if ok:
            st.rerun()  # vuelve a pintar desde session_state (evita duplicar la respuesta)

result = st.session_state.result
if result:
    st.subheader("Respuesta ampliada")
    with st.container(border=True):
        st.markdown(result["text"])
    u = result["usage"]
    if u:
        st.caption(f"Tokens · entrada: {u['prompt_tokens']} · salida: {u['completion_tokens']} · total: {u['total_tokens']}")

    export = {"ocr_text": st.session_state.ocr_text, "response": result["text"],
              "settings": result["settings"], "metrics": st.session_state.metrics}
    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button("Descargar respuesta (.md)", result["text"], "respuesta.md", "text/markdown", width="stretch")
    d2.download_button("Descargar métricas (.json)", json.dumps(export, ensure_ascii=False, indent=2),
                       "resultado.json", "application/json", width="stretch")

    if st.session_state.metrics:
        render_metrics(st.session_state.metrics)
