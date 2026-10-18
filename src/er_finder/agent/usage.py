"""Token and latency accounting shared by the agent graph and the input classifier."""

from dataclasses import dataclass

from langchain_core.callbacks import BaseCallbackHandler


class RunUsage(BaseCallbackHandler):
    """Retain token counts only, never model inputs, outputs, or exceptions."""

    def __init__(self):
        self.responses = 0
        self.known_responses = 0
        self.tokens = dict(input_tokens=0, output_tokens=0, total_tokens=0)

    def on_llm_end(self, response, **kwargs):
        self.responses += 1
        if not response.generations or not response.generations[0]:
            return
        message = getattr(response.generations[0][0], "message", None)
        usage = getattr(message, "usage_metadata", None)
        if not usage or not all(
            type(usage.get(key)) is int and usage[key] >= 0 for key in self.tokens
        ):
            return
        self.known_responses += 1
        for key in self.tokens:
            self.tokens[key] += usage[key]


@dataclass
class ClassifierUsage:
    """One classifier turn; calls is 0 when the rule engine decided without a model."""

    calls: int = 0
    elapsed_seconds: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    tokens_complete: bool = True
