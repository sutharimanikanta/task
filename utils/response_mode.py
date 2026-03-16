def build_prompt(query: str, mode: str) -> str:
    if mode == "Concise":
        return (
            "Answer in 2-3 sentences. Be direct.\n\nQuestion: " + query
        )
    if mode == "Detailed":
        return (
            "Provide a thorough answer with examples where helpful. "
            "Use bullet points or headings if appropriate.\n\nQuestion: " + query
        )
    if mode == "Reasoning":
        return (
            "Think step by step before answering. "
            "First write your reasoning prefixed with 'Thinking:' on separate lines, "
            "then write your final answer prefixed with 'Answer:'.\n\nQuestion: " + query
        )
    return "Question: " + query
