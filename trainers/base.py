from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self, config, model, train_loader, val_loader=None):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    @abstractmethod
    def train(self):
        pass
