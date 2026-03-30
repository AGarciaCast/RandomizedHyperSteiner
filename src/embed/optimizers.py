import torch


class EmbedOptim():
    # An abstract class for optimizers for embedding methods

    def __init__(self, embedder, lr):
        # embedder: Embedder , lr: Float

        self.embedder = embedder
        self.lr = lr

    def zero_grad(self):
        self.embedder.embeddings.grad = torch.zeros_like(self.embedder.embeddings)

    def step(self):
        pass


class StandardOptim(EmbedOptim):
    def step(self, idx):
        with torch.no_grad():
            embeddings = self.embedder.embeddings
            # print(embeddings.grad)
            # embeddings[idx] -= self.lr * self.embedder.embeddings.grad[idx]

            grad = self.embedder.embeddings.grad.detach()
            embeddings[idx] -= self.lr * grad[idx]


class PoincareOptim(EmbedOptim):
    def step(self, idx):
        with torch.no_grad():
            embeddings = self.embedder.embeddings
            norms = (embeddings ** 2).sum(-1).unsqueeze(-1)
            embeddings[idx] -= self.lr * (1 - norms[idx]) ** 2 * embeddings.grad[idx] / 4