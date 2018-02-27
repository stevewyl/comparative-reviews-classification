import json


class ModelConfig(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self, max_words, embed_size, vocab_cnt, model_name):
        self.max_words = max_words
        self.embed_size = embed_size
        self.vocab_cnt = vocab_cnt

        if model_name in ['HAN', 'MHAN', 'Self_Att']:
            if model_name == 'Self_Att':
                self.

        if model_name == 'TextCNN':
            self.



    def save(self, file):
        with open(file, 'w') as f:
            json.dump(vars(self), f, sort_keys=True, indent=4)

    @classmethod
    def load(cls, file):
        with open(file) as f:
            variables = json.load(f)
            self = cls()
            for key, val in variables.items():
                setattr(self, key, val)
        return self


class TrainingConfig(object):
    """Wrapper class for training hyperparameters."""

    def __init__(self, batch_size=64, optimizer='adam', learning_rate=0.001, lr_decay=0.9,
                 clip_gradients=5.0, max_epoch=10, early_stopping=True, patience=2,
                 train_embeddings=False, max_checkpoints_to_keep=5):

        # Batch size
        self.batch_size = batch_size

        # Optimizer for training the model.
        self.optimizer = optimizer

        # Learning rate for the initial phase of training.
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay

        # If not None, clip gradients to this value.
        self.clip_gradients = clip_gradients

        # The number of max epoch size
        self.max_epoch = max_epoch

        # Parameters for early stopping
        self.early_stopping = early_stopping
        self.patience = patience

        # Fine-tune word embeddings
        self.train_embeddings = train_embeddings

        # How many model checkpoints to keep.
        self.max_checkpoints_to_keep = max_checkpoints_to_keep
