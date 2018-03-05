from reader import batch_iter

class Evaluator(object):

    def __init__(self, model):
        self.model = model

    def eval(self, x_test, y_test):
        train_steps, train_batches = batch_iter(x_test,
                                                y_test,
                                                batch_size=64,  # Todo: if batch_size=1, eval does not work.
                                                shuffle=False)
        res = self.model.model.evaluate_generator(generator=train_batches,
                                                  steps=train_steps)
        return res