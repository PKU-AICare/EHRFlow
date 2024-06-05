from langchain_core.callbacks import StreamingStdOutCallbackHandler, CallbackManagerForChainRun
from langchain_core.outputs import LLMResult


class ChainStreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self):
        self.tokens = []
        self.str = ''
        # 记得结束后这里置true
        self.finish = False

    def on_llm_new_token(self, token: str, **kwargs):
        print(token)
        self.str += token
        self.tokens.append(token)

    def on_llm_end(self, response: LLMResult, **kwargs: any) -> None:
        self.finish = 1

    def on_llm_error(self, error: Exception, **kwargs: any) -> None:
        print(str(error))
        self.tokens.append(str(error))

    def generate_tokens(self):
        while not self.finish or self.tokens:
            if self.tokens:
                data = self.tokens.pop(0)
                yield data
            else:
                pass
