from transformers import LlamaConfig


class BranchyLlamaConfig(LlamaConfig):
    def __init__(self, self_supervision=False, **kwargs):
        super().__init__(**kwargs)
        self.self_supervision = self_supervision
