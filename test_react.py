import streamlit as st
import streamlit.components.v1 as components
import sys
import os
sys.path.append(os.path.dirname(__file__))

from router import retrieve_with_fallback
from answer_engine import process

# ── Page config 
st.set_page_config(
    page_title="Document Intelligence",
    page_icon="📄",
    layout="wide"
)

# ── Session state init ────────────────────────────────────────────────────────
if "ingested_doc"   not in st.session_state: st.session_state.ingested_doc   = "1706.03762v7"
if "read_generator" not in st.session_state: st.session_state.read_generator = None
if "current_page"   not in st.session_state: st.session_state.current_page   = None
if "page_content"   not in st.session_state: st.session_state.page_content   = []
if "reading_active" not in st.session_state: st.session_state.reading_active = False
if "chat_history"   not in st.session_state: st.session_state.chat_history   = []
if "last_answer"    not in st.session_state: st.session_state.last_answer    = None

# ── Voice helper ──────────────────────────────────────────────────────────────
def speak(text, language, speed, pitch):
    lang_map = {
        "English": "en-US", "Hindi": "hi-IN", "Telugu": "te-IN",
        "Tamil": "ta-IN", "Spanish": "es-ES", "French": "fr-FR",
        "German": "de-DE", "Arabic": "ar-SA", "Japanese": "ja-JP",
        "Chinese": "zh-CN", "Portuguese": "pt-BR", "Korean": "ko-KR",
        "Italian": "it-IT", "Russian": "ru-RU"
    }
    lang_code = lang_map.get(language, "en-US")
    safe_text = text.replace("'", " ").replace('"', " ").replace("\n", " ").replace("\\", " ")
    js = f"""
    <script>
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance('{safe_text}');
    u.lang  = '{lang_code}';
    u.rate  = {speed};
    u.pitch = {pitch};
    window.speechSynthesis.speak(u);
    </script>
    """
    components.html(js, height=0)

def stop_voice():
    components.html("<script>window.speechSynthesis.cancel();</script>", height=0)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    st.subheader("📂 Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        save_path = os.path.join("./uploaded_docs", uploaded_file.name)
        os.makedirs("./uploaded_docs", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())

        if st.button("📥 Ingest Document"):
            from ingestion import ingest_pdf
            with st.spinner("Ingesting — this may take a few minutes..."):
                ingest_pdf(save_path)
                st.session_state.ingested_doc = os.path.splitext(uploaded_file.name)[0]
            st.success(f"✅ Done: {st.session_state.ingested_doc}")

    st.divider()
    manual_doc = st.text_input("Document name (without .pdf)", value=st.session_state.ingested_doc)
    if manual_doc:
        st.session_state.ingested_doc = manual_doc

    st.divider()
    st.subheader("🌐 Language")
    language = st.selectbox("Response language", [
        "English", "Hindi", "Telugu", "Tamil", "Spanish",
        "French", "German", "Arabic", "Japanese", "Chinese",
        "Portuguese", "Korean", "Italian", "Russian"
    ])

    st.subheader("📖 Style")
    style = st.radio("Explanation style", ["simple", "technical", "eli5"],
        format_func=lambda x: {"simple": "Simple", "technical": "Technical", "eli5": "ELI5 (Like I'm 5)"}[x])

    st.subheader("📄 Reading mode")
    granularity = st.radio("When reading full document", ["paragraph", "page"],
        format_func=lambda x: {"paragraph": "Paragraph by paragraph", "page": "Full page at once"}[x])

    st.divider()
    st.subheader("🔊 Voice")
    voice_on    = st.toggle("Enable voice output", value=False)
    voice_speed = st.slider("Speed", 0.5, 2.0, 1.0, 0.1) if voice_on else 1.0
    voice_pitch = st.slider("Pitch", 0.5, 2.0, 1.0, 0.1) if voice_on else 1.0

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📄 Multimodal Document Intelligence")
st.caption("Ask questions, read documents, understand charts and tables — in any language.")

if not st.session_state.ingested_doc:
    st.info("👈 Upload and ingest a PDF to get started.")
    st.stop()

st.success(f"📗 Active document: **{st.session_state.ingested_doc}**")

tab1, tab2 = st.tabs(["💬 Ask a Question", "📖 Read Document"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Question answering
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Ask anything about the document")

    question = st.text_input(
        "Your question",
        placeholder="e.g. What BLEU score did Transformer big achieve?",
        key="question_input"
    )

    # Button is always enabled — validation happens after click
    if st.button("🔍 Get Answer", key="answer_btn"):
        if not question.strip():
            st.warning("Please type a question first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    intent, result = process(
                        question      = question,
                        document_name = st.session_state.ingested_doc,
                        language      = language,
                        style         = style,
                        granularity   = granularity,
                        router_func   = retrieve_with_fallback
                    )
                    st.session_state.last_answer = result
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer"  : result.get("answer", ""),
                        "page"    : result.get("page_number"),
                        "modality": result.get("modality")
                    })
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.last_answer = None

    # Display last answer
    if st.session_state.last_answer:
        result = st.session_state.last_answer
        st.markdown("### Answer")
        st.markdown(result.get("answer", ""))

        col1, col2, col3 = st.columns(3)
        col1.metric("Page",     result.get("page_number", "—"))
        col2.metric("Modality", str(result.get("modality", "—")).upper())
        col3.metric("Language", result.get("language", "—"))

        if voice_on:
            speak(result.get("answer", ""), language, voice_speed, voice_pitch)
            if st.button("⏹ Stop voice", key="stop_q"):
                stop_voice()

    # Chat history
    if st.session_state.chat_history:
        st.divider()
        st.subheader("History")
        for item in reversed(st.session_state.chat_history):
            with st.expander(f"Q: {item['question'][:70]}"):
                st.markdown(item["answer"])
                st.caption(f"Page {item['page']} | {str(item['modality']).upper()}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Full document reading
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Read the full document")
    st.caption(f"Mode: **{granularity}** | Language: **{language}** | Style: **{style}**")

    col_start, col_stop = st.columns([1, 1])

    with col_start:
        if st.button("▶ Start Reading", key="start_btn"):
            with st.spinner("Preparing document..."):
                try:
                    intent, generator = process(
                        question      = "read the document",
                        document_name = st.session_state.ingested_doc,
                        language      = language,
                        style         = style,
                        granularity   = granularity,
                        router_func   = retrieve_with_fallback
                    )
                    page_num, page_results             = next(generator)
                    st.session_state.read_generator    = generator
                    st.session_state.current_page      = page_num
                    st.session_state.page_content      = page_results
                    st.session_state.reading_active    = True
                except StopIteration:
                    st.warning("No content found in document.")
                except Exception as e:
                    st.error(f"Error starting read: {e}")
            st.rerun()

    with col_stop:
        if st.button("⏹ Stop Reading", key="stop_btn"):
            st.session_state.reading_active = False
            st.session_state.read_generator = None
            st.session_state.page_content   = []
            stop_voice()
            st.rerun()

    # Display current page
    if st.session_state.reading_active and st.session_state.page_content:
        st.divider()
        st.markdown(f"### 📄 Page {st.session_state.current_page}")

        full_text = ""
        for item in st.session_state.page_content:
            icon = {"text": "📝", "table": "📊", "chart": "📈"}.get(item.get("modality"), "📄")
            st.markdown(f"{icon} {item['answer']}")
            st.divider()
            full_text += item.get("answer", "") + " "

        if voice_on and full_text.strip():
            speak(full_text, language, voice_speed, voice_pitch)

        col_cont, col_info = st.columns([1, 3])
        with col_cont:
            if st.button("▶ Continue to next page", key="continue_btn"):
                stop_voice()
                try:
                    page_num, page_results            = next(st.session_state.read_generator)
                    st.session_state.current_page     = page_num
                    st.session_state.page_content     = page_results
                except StopIteration:
                    st.session_state.reading_active   = False
                    st.session_state.read_generator   = None
                    st.success("✅ Finished reading the document.")
                st.rerun()

        with col_info:
            st.caption(f"Page {st.session_state.current_page} of '{st.session_state.ingested_doc}'")