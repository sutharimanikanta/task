import os
import sys
import time

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import AVAILABLE_MODELS, SMALL_MODEL
from models.llm import get_model
from utils.context_builder import build_context
from utils.discriminator import validate_response
from utils.memory import maybe_summarize
from utils.query_rewriter import rewrite_query
from utils.rag import build_retriever
from utils.response_mode import build_prompt
from utils.router import is_chitchat

st.set_page_config(
    page_title="The Chatbot Blueprint",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_retriever(files):
    return build_retriever(files)


def stream_text(text: str):
    for word in text.split():
        yield word + " "
        time.sleep(0.025)


def get_llm_response(chat_model, messages: list, system_prompt: str) -> str:
    try:
        formatted = [SystemMessage(content=system_prompt)]
        for m in messages:
            if m["role"] == "user":
                formatted.append(HumanMessage(content=m["content"]))
            else:
                formatted.append(AIMessage(content=m["content"]))
        return chat_model.invoke(formatted).content
    except Exception as e:
        return f"Error: {str(e)}"


def render_response(response: str, mode: str):
    if mode == "Reasoning" and "Thinking:" in response and "Answer:" in response:
        parts = response.split("Answer:", 1)
        thinking = parts[0].replace("Thinking:", "").strip()
        answer = parts[1].strip()
        with st.expander("Chain of thought"):
            st.markdown(thinking)
        st.write_stream(stream_text(answer))
        return answer
    st.write_stream(stream_text(response))
    return response


def render_trail(trail: list):
    if not trail:
        return
    lines = []
    for step in trail:
        icon = "found" if step["found"] else "not found"
        lines.append(f"{step['source']} -> {icon} | query: _{step['sub_query']}_")
    with st.expander("Source trail"):
        st.markdown("\n\n".join(lines))


def prompt_rewrite_widget(original: str, model):
    """
    Shows the rewritten query in an editable text input.
    User can confirm the rewrite, edit it, or skip and use the original.
    Returns (final_query, should_proceed).
    """
    rewritten = rewrite_query(original, model)
    st.markdown("Query rewriter — edit or skip:")
    edited = st.text_input(
        "Rewritten query",
        value=rewritten,
        key="rewritten_query_input",
        label_visibility="collapsed",
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        confirmed = st.button(
            "Use rewritten query", key="confirm_rewrite", use_container_width=True
        )
    with col2:
        skipped = st.button(
            "Skip — use original", key="skip_rewrite", use_container_width=True
        )
    if confirmed:
        return edited, True
    if skipped:
        return original, True
    return None, False


def chat_page():
    mode = st.session_state.get("mode", "Concise")
    model_name = st.session_state.get("model", AVAILABLE_MODELS[0])
    persona = st.session_state.get("persona", "You are a helpful AI assistant.")

    chat_model = get_model(model_name)
    small_model = get_model(SMALL_MODEL)

    if chat_model is None:
        st.error("GROQ_API_KEY not found. Add it to config/.env and restart.")
        return

    uploaded_files = st.file_uploader(
        "Upload documents (PDF or DOCX)",
        accept_multiple_files=True,
        type=["pdf", "docx"],
    )

    retriever = None
    if uploaded_files:
        with st.spinner("Indexing documents..."):
            retriever = load_retriever(tuple(uploaded_files))
        if retriever:
            st.success(f"{len(uploaded_files)} file(s) indexed.")
        else:
            st.warning("Could not index the uploaded files.")

    has_docs = retriever is not None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and "Thinking:" in msg.get("content", ""):
                parts = msg["content"].split("Answer:", 1)
                with st.expander("Chain of thought"):
                    st.markdown(parts[0].replace("Thinking:", "").strip())
                st.markdown(parts[1].strip() if len(parts) > 1 else "")
            else:
                st.markdown(msg["content"])
            if msg.get("trail"):
                render_trail(msg["trail"])

    raw_input = st.chat_input("Ask anything...")

    if raw_input:
        st.session_state.pending_query = raw_input
        st.session_state.rewrite_confirmed = False
        st.rerun()

    if st.session_state.pending_query and not st.session_state.get("rewrite_confirmed"):
        with st.container():
            st.info(f"Original: {st.session_state.pending_query}")
            final_query, proceed = prompt_rewrite_widget(
                st.session_state.pending_query,
                small_model or chat_model,
            )
            if proceed:
                st.session_state.final_query = final_query
                st.session_state.rewrite_confirmed = True
                st.rerun()
        return

    if st.session_state.get("rewrite_confirmed") and st.session_state.get(
        "final_query"
    ):
        prompt = st.session_state.final_query
        original = st.session_state.pending_query

        st.session_state.pending_query = None
        st.session_state.rewrite_confirmed = False
        st.session_state.final_query = None

        st.session_state.messages.append({"role": "user", "content": original})
        with st.chat_message("user"):
            st.markdown(original)

        with st.chat_message("assistant"):
            with st.spinner(""):
                if is_chitchat(prompt):
                    context = ""
                    sub_queries = [prompt]
                    trail = []
                else:
                    context, sub_queries, trail = build_context(
                        prompt, small_model or chat_model, retriever, has_docs
                    )

                # Override "Reasoning" mode for simple queries - use "Concise" instead
                effective_mode = mode
                if mode == "Reasoning" and not is_query_complex(prompt):
                    effective_mode = "Concise"

                final_query_prompt = build_prompt(prompt, effective_mode)

                system_prompt = persona
                if context:
                    system_prompt += (
                        "\n\nUse the context below. If it does not fully answer the question, "
                        "use your own knowledge.\n\nContext:\n" + context
                    )
                system_prompt += "\n\n" + final_query_prompt

                history = maybe_summarize(st.session_state.messages, chat_model)
                response = get_llm_response(chat_model, history, system_prompt)

                is_valid = True
                if not is_chitchat(prompt) and small_model:
                    is_valid, reason = validate_response(prompt, response, small_model)
                    if not is_valid:
                        from utils.web_search import web_search as _ws

                        fallback_ctx, fallback_urls = _ws(prompt)
                        if fallback_ctx:
                            system_prompt = (
                                persona
                                + "\n\nContext:\n"
                                + fallback_ctx
                                + "\n\n"
                                + final_query_prompt
                            )
                            response = get_llm_response(
                                chat_model, history, system_prompt
                            )
                            trail.append(
                                {
                                    "sub_query": prompt,
                                    "source": f"discriminator fallback: Web ({fallback_urls})",
                                    "found": True,
                                }
                            )

                display_response = render_response(response, effective_mode)

            render_trail(trail)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "trail": trail,
            }
        )


def instructions_page():
    st.title("Setup Guide")
    st.markdown("""
## Installation

```bash
pip install -r requirements.txt
```

## API keys

Create `config/.env`:

```
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
```

## Pipeline

```
User query
  -> Chitchat check (skip retrieval entirely if greeting)
  -> Query rewriter widget (user can edit before submitting)
  -> Query decomposition (complex queries split into sub-questions)
  -> Per sub-question: RAG with threshold + reranker
       -> fallback to web search if RAG returns nothing
  -> All contexts merged
  -> Main LLM generates answer
  -> Discriminator validates answer (small model)
       -> fallback to web search if answer invalid
  -> Source trail shown under response
```
    """)


def main():
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("", ["Chat", "Instructions"], index=0)

        st.divider()
        st.subheader("Model")
        model = st.selectbox("", AVAILABLE_MODELS, index=0)
        st.session_state["model"] = model

        st.subheader("Response mode")
        mode = st.selectbox("", ["Concise", "Detailed", "Reasoning"], index=0)
        st.session_state["mode"] = mode

        st.subheader("Persona")
        persona = st.text_area("", value="You are a helpful AI assistant.", height=80)
        st.session_state["persona"] = persona

        if page == "Chat" and st.session_state.get("messages"):
            st.divider()
            history_text = "\n\n".join(
                f"{m['role'].upper()}: {m['content']}"
                for m in st.session_state.messages
            )
            st.download_button(
                "Export chat",
                data=history_text,
                file_name="chat_history.txt",
                mime="text/plain",
                use_container_width=True,
            )
            if st.button("Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    if page == "Instructions":
        instructions_page()
    else:
        st.title("AI Chatbot")
        chat_page()


if __name__ == "__main__":
    main()
