# OCR + ampliación con LLM (Streamlit)

App de Streamlit: sube una imagen → OCR extrae el texto → un LLM de OpenAI lo amplía en tono **formal** o **técnico** → métricas de calidad del texto generado.

## Uso en la app
1. Ingresa tu **API key de OpenAI** en la barra lateral (no se guarda; vive solo en la sesión).
2. Sube una imagen y pulsa **Extraer texto (OCR)**. El texto es editable.
3. Elige estilo (Formal / Técnica), extensión e idioma, ajusta los parámetros del LLM en la barra lateral y pulsa **Generar respuesta ampliada**.
4. Revisa las métricas y descarga la respuesta (.md) o el resultado completo (.json).

## Despliegue en Streamlit Community Cloud
1. Sube este repositorio a GitHub (rama `main`).
2. En [share.streamlit.io](https://share.streamlit.io) → **Create app** → elige el repo, rama `main` y archivo principal `app.py`.
3. En **Advanced settings** selecciona Python 3.11 o 3.12.
4. **Deploy.** `packages.txt` instala Tesseract (español e inglés) y `requirements.txt` las librerías de Python. No hace falta configurar *secrets*: cada usuario pone su propia key en la interfaz.

> No definas tu API key en *Secrets* ni como variable de entorno si la app es pública: la usarían todos los visitantes con tu cuenta.

### Extras opcionales
Están en `requirements-extras.txt` con instrucciones. Cópialos a `requirements.txt` para activarlos:
- **spaCy** (profundidad sintáctica y POS). Los modelos se instalan por URL, sin comandos adicionales.
- **LanguageTool** (gramática objetiva). Además agrega `default-jre-headless` a `packages.txt`.

Sin extras, la app funciona igual usando métricas locales y el evaluador LLM.

## Métricas

| Bloque | Qué mide | Cómo |
|---|---|---|
| Estadísticas | palabras, oraciones, párrafos, diversidad léxica (MATTR), repetición de trigramas, legibilidad | local. Szigriszt-Pazos (INFLESZ) en español, Flesch en inglés |
| Sintaxis | longitud y variación de oraciones, comas, subordinantes, conectores; con spaCy: profundidad del árbol y verbos/sustantivos | local |
| Semántica | similitud OCR ↔ respuesta, cobertura del OCR, contenido añadido | embeddings + coseno |
| Coherencia | local (oraciones consecutivas), global (vs. centroide), transición más débil | embeddings + coseno |
| Gramática | puntajes 0-100 (gramática, sintaxis, cohesión, claridad), errores señalados, afirmaciones a verificar | evaluador LLM; opcional LanguageTool |

Los puntajes del evaluador LLM son subjetivos y el coseno depende del modelo de embeddings: úsalos para **comparar** respuestas (Formal vs. Técnica, distintas temperaturas), no como umbrales absolutos. El conteo de sílabas en español es heurístico.

## Costos y privacidad
- La imagen se procesa en el servidor de la app con Tesseract; solo el **texto extraído** y la respuesta se envían a OpenAI.
- Cada generación hace 3 llamadas: chat, embeddings y evaluador. Desactiva las dos últimas en «Métricas» para gastar menos.
- Modelos de razonamiento (o-series, gpt-5): no aceptan `temperature`/`top_p`; sube «Máx. tokens» a 4000 o más.

## Estructura
```
app.py               # interfaz Streamlit
src/ocr.py           # preprocesado y OCR
src/llm.py           # prompts y llamadas a OpenAI
src/metrics.py       # métricas
packages.txt         # dependencias del sistema (Tesseract)
requirements.txt     # dependencias de Python
.streamlit/config.toml
tests/               # pytest (opcional, sin API key)
```
