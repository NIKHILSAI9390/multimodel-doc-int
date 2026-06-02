import streamlit as st
import streamlit.components.v1 as components
import sys
import os
import re
import json
sys.path.append(os.path.dirname(__file__))

from router import retrieve_with_fallback
from answer_engine import process, build_system_prompt
import anthropic
import chromadb

# ── Config 
# CLAUDE_API_KEY = ""
client    = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
chroma    = chromadb.PersistentClient(path="./chroma_db")
text_col  = chroma.get_or_create_collection("text_chunks")
table_col = chroma.get_or_create_collection("table_chunks")
chart_col = chroma.get_or_create_collection("chart_chunks")

st.set_page_config(page_title="Document Intelligence", page_icon="📄", layout="wide")

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "ingested_doc"    : "1706.03762v7",
    "chat_history"    : [],
    "last_answer"     : None,
    "doc_paragraphs"  : [],
    "current_para_idx": 0,
    "is_reading"      : False,
    "explanation"     : "",
    "lang_summary"    : "",
    "active_language" : "English",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

LANG_MAP = {
    "English":"en-US","Hindi":"hi-IN","Telugu":"te-IN","Tamil":"ta-IN",
    "Spanish":"es-ES","French":"fr-FR","German":"de-DE","Arabic":"ar-SA",
    "Japanese":"ja-JP","Chinese":"zh-CN","Portuguese":"pt-BR","Korean":"ko-KR",
    "Italian":"it-IT","Russian":"ru-RU"
}

def clean_for_voice(text):
    text = re.sub(r"#{1,6}\s*","",text)
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}",r"\1",text)
    text = re.sub(r"`[^`]*`","",text)
    text = re.sub(r"^\s*[-*+]\s+","",text,flags=re.MULTILINE)
    text = re.sub(r"\n+",". ",text)
    return re.sub(r"\s{2,}"," ",text).strip()

def speak(text, language, speed, pitch):
    lc   = LANG_MAP.get(language,"en-US")
    safe = clean_for_voice(text).replace("'"," ").replace('"'," ").replace("\\"," ")
    components.html(f"""<script>
    function go(){{
        speechSynthesis.cancel();
        var u=new SpeechSynthesisUtterance('{safe}');
        u.lang='{lc}';u.rate={speed};u.pitch={pitch};
        var m=speechSynthesis.getVoices().find(v=>v.lang.startsWith('{lc[:2]}'));
        if(m)u.voice=m;speechSynthesis.speak(u);
    }}
    speechSynthesis.getVoices().length===0?(speechSynthesis.onvoiceschanged=go):go();
    </script>""", height=0)

def stop_voice():
    components.html("<script>speechSynthesis.cancel();</script>",height=0)

def load_paragraphs(doc_name):
    items=[]
    r=text_col.get(where={"document_name":doc_name},include=["documents","metadatas"])
    for doc,meta in zip(r["documents"],r["metadatas"]):
        for i,p in enumerate([x.strip() for x in doc.split("\n\n") if len(x.strip())>40]):
            items.append({"id":f"p{meta['page_number']}_t{i}","text":p,"page":meta["page_number"],"modality":"text","sub":i})
    r=table_col.get(where={"document_name":doc_name},include=["documents","metadatas"])
    for doc,meta in zip(r["documents"],r["metadatas"]):
        items.append({"id":f"p{meta['page_number']}_tbl","text":doc,"page":meta["page_number"],"modality":"table","sub":0})
    r=chart_col.get(where={"document_name":doc_name},include=["documents","metadatas"])
    for doc,meta in zip(r["documents"],r["metadatas"]):
        items.append({"id":f"p{meta['page_number']}_chr","text":doc,"page":meta["page_number"],"modality":"chart","sub":0})
    items.sort(key=lambda x:(x["page"],{"text":0,"table":1,"chart":2}[x["modality"]],x["sub"]))
    return items

def explain_para(text, modality, language, style):
    system = build_system_prompt(language, style)
    icons  = {"text":"paragraph","table":"data table","chart":"figure or chart"}
    resp   = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=512, system=system,
        messages=[{"role":"user","content":f"Explain this {icons.get(modality,'content')} clearly:\n\n{text}"}]
    )
    return resp.content[0].text.strip()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    st.subheader("📂 Document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded:
        path = os.path.join("./uploaded_docs", uploaded.name)
        os.makedirs("./uploaded_docs", exist_ok=True)
        open(path,"wb").write(uploaded.read())
        if st.button("📥 Ingest"):
            from ingestion import ingest_pdf
            with st.spinner("Ingesting..."):
                ingest_pdf(path)
                st.session_state.ingested_doc    = os.path.splitext(uploaded.name)[0]
                st.session_state.doc_paragraphs  = []
            st.success("✅ Done")

    st.divider()
    doc = st.text_input("Document name (no .pdf)", value=st.session_state.ingested_doc)
    if doc != st.session_state.ingested_doc:
        st.session_state.ingested_doc   = doc
        st.session_state.doc_paragraphs = []

    st.divider()
    language = st.selectbox("🌐 Language", list(LANG_MAP.keys()))

    # Clear summary when language changes
    if language != st.session_state.get("active_language", "English"):
        st.session_state.lang_summary    = ""
        st.session_state.active_language = language

    style    = st.radio("📖 Style", ["simple","technical","eli5"],
                 format_func=lambda x:{"simple":"Simple","technical":"Technical","eli5":"ELI5"}[x])
    st.divider()
    voice_on    = st.toggle("🔊 Voice output", value=False)
    voice_speed = st.slider("Speed",0.5,2.0,1.0,0.1) if voice_on else 1.0
    voice_pitch = st.slider("Pitch",0.5,2.0,1.0,0.1) if voice_on else 1.0

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📄 Multimodal Document Intelligence")
st.caption("Ask questions · Read documents · Understand charts and tables · In any language")
st.success(f"📗 Active: **{st.session_state.ingested_doc}**")

tab1, tab2, tab3 = st.tabs(["💬 Ask a Question","📖 Read Document","🔍 Selected Text"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Ask anything about the document")
    q = st.text_input("Your question", placeholder="e.g. What BLEU score did Transformer big achieve?", key="q")
    if st.button("🔍 Get Answer"):
        if not q.strip():
            st.warning("Type a question first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    _, result = process(q, st.session_state.ingested_doc, language, style, "paragraph", retrieve_with_fallback)
                    st.session_state.last_answer = result
                    st.session_state.chat_history.append({"question":q,"answer":result.get("answer",""),"page":result.get("page_number"),"modality":result.get("modality")})
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.last_answer:
        r = st.session_state.last_answer
        st.markdown("### Answer")
        st.markdown(r.get("answer",""))
        c1,c2,c3 = st.columns(3)
        c1.metric("Page", r.get("page_number","—"))
        c2.metric("Modality", str(r.get("modality","—")).upper())
        c3.metric("Language", r.get("language","—"))
        if voice_on:
            speak(r.get("answer",""), language, voice_speed, voice_pitch)
            if st.button("⏹ Stop", key="sq"): stop_voice()

    if st.session_state.chat_history:
        st.divider()
        st.subheader("History")
        for item in reversed(st.session_state.chat_history):
            with st.expander(f"Q: {item['question'][:70]}"):
                st.markdown(item["answer"])
                st.caption(f"Page {item['page']} | {str(item['modality']).upper()}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Document viewer
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    # Load on first visit
    if not st.session_state.doc_paragraphs:
        with st.spinner("Loading document content..."):
            st.session_state.doc_paragraphs   = load_paragraphs(st.session_state.ingested_doc)
            st.session_state.current_para_idx = 0
            st.session_state.explanation      = ""

    paras = st.session_state.doc_paragraphs
    total = len(paras)

    if not paras:
        st.warning("No content found. Ingest a document first.")
    else:
        idx  = st.session_state.current_para_idx
        cur  = paras[idx]

        # Progress bar
        st.progress((idx+1)/total, text=f"Section {idx+1} of {total} | Page {cur['page']} | {cur['modality'].capitalize()}")

        left, right = st.columns([1,1])

        # ── LEFT: Document viewer ─────────────────────────────────────────────
        with left:
            st.subheader("📄 Document View")
            st.caption("Click any paragraph to jump there. Currently highlighted section is active.")

            # Build clickable document HTML
            html = ["<div style='font-family:Georgia,serif;font-size:13px;line-height:1.7;height:580px;overflow-y:auto;padding:8px 12px;background:#111;border-radius:8px;'>"]
            cur_page = None
            for i, p in enumerate(paras):
                if p["page"] != cur_page:
                    cur_page = p["page"]
                    html.append(f"<div style='color:#666;font-size:11px;text-align:center;margin:14px 0 6px;letter-spacing:2px;'>── PAGE {cur_page} ──</div>")

                is_cur = (i == idx)
                icon   = {"table":"📊 ","chart":"📈 "}.get(p["modality"],"")
                bg     = "#0d3a5c" if is_cur else "transparent"
                border = "left:3px solid #4a9eff" if is_cur else "left:3px solid transparent"
                color  = "#fff" if is_cur else "#aaa"
                preview= p["text"][:180] + ("..." if len(p["text"])>180 else "")

                html.append(
                    f"<div id='s{i}' "
                    f"style='background:{bg};border-{border};padding:7px 10px;margin:3px 0;cursor:pointer;color:{color};border-radius:4px;' "
                    f"onclick=\"document.querySelectorAll('[data-idx]').forEach(e=>e.remove());var h=document.createElement('input');h.type='hidden';h.setAttribute('data-idx','{i}');h.id='clicked';document.body.appendChild(h);window.parent.postMessage({{paraIdx:{i}}},'*');\" "
                    f"onmouseover=\"if(!{str(is_cur).lower()})this.style.background='#1a2d3d'\" "
                    f"onmouseout=\"if(!{str(is_cur).lower()})this.style.background='transparent'\">"
                    f"{icon}{preview}</div>"
                )

            # Auto-scroll to current
            html.append(f"<script>var e=document.getElementById('s{idx}');if(e)e.scrollIntoView({{block:'center',behavior:'smooth'}});</script>")
            html.append("</div>")

            components.html("".join(html), height=600, scrolling=False)

            # Paragraph jump input — user types section number
            st.divider()
            jump = st.number_input("Jump to section", min_value=1, max_value=total, value=idx+1, step=1, key="jump_input")
            if st.button("Go", key="jump_btn"):
                st.session_state.current_para_idx = int(jump) - 1
                st.session_state.explanation      = ""
                st.rerun()

        # ── RIGHT: Explanation panel ──────────────────────────────────────────
        with right:
            st.subheader("💡 Explanation")
            st.caption(f"📍 Page {cur['page']} · {cur['modality'].capitalize()} · Section {idx+1}/{total}")

            with st.expander("Original text", expanded=False):
                st.markdown(cur["text"])

            # ── Navigation: Prev / Next / Stop ───────────────────────────────
            n1, n2, n3 = st.columns(3)
            with n1:
                if st.button("⬅ Prev", key="prev_btn"):
                    if idx > 0:
                        st.session_state.current_para_idx = idx - 1
                        st.session_state.explanation  = ""
                        st.session_state.lang_summary = ""
                        stop_voice()
                    st.rerun()
            with n2:
                if st.button("➡ Next", key="next_nav"):
                    if idx + 1 < total:
                        st.session_state.current_para_idx = idx + 1
                        st.session_state.explanation  = ""
                        st.session_state.lang_summary = ""
                    st.rerun()
            with n3:
                if st.button("⏹ Stop voice", key="stop_nav"):
                    stop_voice()

            st.divider()

            # ── Three action buttons ──────────────────────────────────────────
            a1, a2, a3 = st.columns(3)

            # Button 1 — reads original text in English (document language)
            with a1:
                if st.button("🔊 Read original", key="read_orig"):
                    if not voice_on:
                        st.warning("Enable voice in sidebar.")
                    else:
                        speak(cur["text"], "English", voice_speed, voice_pitch)

            # Button 2 — summarise in selected language and read it aloud
            with a2:
                btn_label = f"🌐 {language} summary" if language != "English" else "🌐 Summary"
                if st.button(btn_label, key="sum_lang"):
                    # Generate only if not already done for this section
                    if not st.session_state.lang_summary:
                        with st.spinner(f"Summarising in {language}..."):
                            try:
                                resp = client.messages.create(
                                    model="claude-haiku-4-5-20251001",
                                    max_tokens=600,
                                    system=(
                                        f"You are a document summarizer. "
                                        f"You must respond entirely in {language}. "
                                        f"Do not write a single word in English."
                                    ),
                                    messages=[{"role":"user","content":
                                        f"Write a detailed summary in {language}. "
                                        f"Cover all key ideas, data, and concepts. "
                                        f"Write 4 to 6 sentences.\n\n{cur['text']}"
                                    }]
                                )
                                st.session_state.lang_summary = resp.content[0].text.strip()
                            except Exception as e:
                                st.error(f"Error: {e}")

                    # Read it aloud in selected language
                    if st.session_state.lang_summary and voice_on:
                        lc   = LANG_MAP.get(language, "en-US")
                        safe = clean_for_voice(st.session_state.lang_summary).replace("'"," ").replace('"'," ")
                        components.html(f"""<script>
                        function readLang(){{
                            speechSynthesis.cancel();
                            var u=new SpeechSynthesisUtterance('{safe}');
                            u.lang='{lc}';u.rate={voice_speed};u.pitch={voice_pitch};
                            var voices=speechSynthesis.getVoices();
                            var match=voices.find(v=>v.lang==='{lc}') ||
                                      voices.find(v=>v.lang.startsWith('{lc[:2]}'));
                            if(match) u.voice=match;
                            speechSynthesis.speak(u);
                        }}
                        speechSynthesis.getVoices().length===0
                            ?(speechSynthesis.onvoiceschanged=readLang):readLang();
                        </script>""", height=0)

            # Button 3 — deep explanation using selected style
            with a3:
                if st.button("💬 Explain", key="exp_btn"):
                    with st.spinner("Explaining..."):
                        try:
                            st.session_state.explanation = explain_para(
                                cur["text"], cur["modality"], language, style
                            )
                        except Exception as e:
                            st.error(f"Error: {e}")

            # ── Show summary (written, no auto-voice) ─────────────────────────
            if st.session_state.lang_summary:
                st.markdown(f"**📝 Summary in {language}:**")
                st.markdown(st.session_state.lang_summary)

            # ── Show explanation ──────────────────────────────────────────────
            if st.session_state.explanation:
                st.markdown("#### Explanation")
                st.markdown(st.session_state.explanation)
                if voice_on:
                    speak(st.session_state.explanation, language, voice_speed, voice_pitch)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Selected text
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Ask about a specific passage")
    st.caption("Paste any text from the document and ask about it or read it aloud.")

    sel_text = st.text_area("Paste the text", height=150,
        placeholder="Paste any paragraph, sentence, or table row here...", key="sel_text")
    sel_q    = st.text_input("What do you want to know?",
        placeholder="Summarize this. / Explain simply. / What does this mean?", key="sel_q")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("💬 Ask", key="sel_ask"):
            if not sel_text.strip(): st.warning("Paste text first.")
            elif not sel_q.strip():  st.warning("Type a question.")
            else:
                with st.spinner("Thinking..."):
                    try:
                        resp = client.messages.create(
                            model="claude-sonnet-4-6", max_tokens=1024,
                            system=build_system_prompt(language, style),
                            messages=[{"role":"user","content":f"Text:\n{sel_text}\n\nQuestion: {sel_q}"}]
                        )
                        ans = resp.content[0].text.strip()
                        st.markdown("### Answer")
                        st.markdown(ans)
                        if voice_on: speak(ans, language, voice_speed, voice_pitch)
                    except Exception as e:
                        st.error(f"Error: {e}")
    with c2:
        if st.button("🔊 Read aloud", key="sel_read"):
            if not sel_text.strip():  st.warning("Paste text first.")
            elif not voice_on:        st.warning("Enable voice in sidebar.")
            else:                     speak(sel_text, language, voice_speed, voice_pitch)

    if st.button("⏹ Stop voice", key="sel_stop"):
        stop_voice()