class BaseTrainer:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("No __init__")

    def dump_logits(self, step=None, eval=True):
        raise NotImplementedError("No dump_logits")

    def train(self, continue_training=False):
        raise NotImplementedError("No train")

    def get_input_pair(self):
        raise NotImplementedError("No get_input_pair")

    def save(self, simple_save=True, ckpt_save=True):
        raise NotImplementedError("No save")

