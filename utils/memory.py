import logging


def maybe_summarize(messages: list, chat_model, threshold: int = 12) -> list:
    """
    When message count exceeds threshold, compress older turns into a
    single summary message to preserve context window space.
    Returns the (possibly compressed) message list.
    """
    if len(messages) <= threshold:
        return messages

    older = messages[:-6]
    recent = messages[-6:]

    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in older
    )
    prompt = (
        "Summarize the following conversation in 3-4 sentences, "
        "keeping the key facts and decisions:\n\n" + history_text
    )
    try:
        from langchain_core.messages import HumanMessage
        res = chat_model.invoke([HumanMessage(content=prompt)])
        summary = {"role": "assistant", "content": "[Summary] " + res.content.strip()}
        return [summary] + recent
    except Exception:
        logging.exception("Memory summarization failed.")
        return messages
