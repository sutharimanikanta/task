import logging


def validate_response(query: str, response: str, small_model) -> tuple[bool, str]:
    """
    Asks a small model to check whether the response actually answers
    the query. Returns (is_valid, reason).
    """
    prompt = (
        f"Does the following answer actually address the question?\n\n"
        f"Question: {query}\n\n"
        f"Answer: {response}\n\n"
        "Reply with exactly one word: YES or NO, then a dash, then one short reason. "
        "Example: YES - answer is relevant. or NO - answer is off-topic."
    )
    try:
        from langchain_core.messages import HumanMessage
        result = small_model.invoke([HumanMessage(content=prompt)])
        text = result.content.strip()
        is_valid = text.upper().startswith("YES")
        reason = text.split("-", 1)[-1].strip() if "-" in text else text
        return is_valid, reason
    except Exception:
        logging.exception("Discriminator failed.")
        return True, "validation skipped"
