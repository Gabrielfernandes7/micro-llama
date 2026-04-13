# src/micro_llama/model.py

import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """Implementa o mecanismo de Multi-head Self-Attention com máscara causal.

    Esta classe projeta os inputs em Query, Key e Value, divide-os em múltiplas 
    cabeças e calcula a atenção ponderada, garantindo que tokens futuros não 
    sejam acessados (causalidade).

    Attributes:
        embed_size (int): Dimensão total do embedding de entrada.
        heads (int): Número de cabeças de atenção paralelas.
        head_dim (int): Dimensão de cada cabeça individual (embed_size // heads).
        values (nn.Linear): Projeção linear para os valores.
        keys (nn.Linear): Projeção linear para as chaves.
        queries (nn.Linear): Projeção linear para as queries.
        fc_out (nn.Linear): Camada linear final de saída.
    """

    def __init__(self, embed_size, heads):
        """Inicializa o mecanismo de atenção.

        Args:
            embed_size (int): Tamanho do vetor de embedding.
            heads (int): Quantidade de cabeças (deve ser divisor de embed_size).
        """
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size precisa ser divisível por heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        """Executa o passo de forward da atenção.

        Args:
            x (torch.Tensor): Tensor de entrada de formato (N, seq_len, embed_size).

        Returns:
            torch.Tensor: Saída da atenção com formato (N, seq_len, embed_size).
        """
        N, seq_len, embed_size = x.shape

        V = self.values(x)
        K = self.keys(x)
        Q = self.queries(x)

        # Dividir em múltiplas cabeças
        V = V.view(N, seq_len, self.heads, self.head_dim)
        K = K.view(N, seq_len, self.heads, self.head_dim)
        Q = Q.view(N, seq_len, self.heads, self.head_dim)

        # Produto escalar (Energy)
        energy = torch.einsum("nqhd,nkhd->nhqk", Q, K)

        # Máscara causal (impedir que o modelo olhe para o futuro)
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        energy = energy.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(energy / (self.embed_size ** 0.5), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", attention, V)
        out = out.reshape(N, seq_len, embed_size)

        return self.fc_out(out)


class TransformerBlock(nn.Module):
    """Bloco fundamental do Transformer contendo Atenção e Feed Forward.

    Aplica normalização de camada (LayerNorm) e conexões residuais (Skip Connections)
    em volta do mecanismo de Self-Attention e da rede Feed Forward.

    Attributes:
        attention (SelfAttention): O módulo de atenção multi-cabeça.
        norm1 (nn.LayerNorm): Normalização após a atenção.
        norm2 (nn.LayerNorm): Normalização após o feed forward.
        feed_forward (nn.Sequential): Rede neural simples de duas camadas lineares.
    """

    def __init__(self, embed_size, heads, forward_expansion):
        """Inicializa o bloco do Transformer.

        Args:
            embed_size (int): Dimensão do embedding.
            heads (int): Número de cabeças de atenção.
            forward_expansion (int): Fator de expansão da camada oculta do feed forward.
        """
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

    def forward(self, x):
        """Executa o forward do bloco com conexões residuais.

        Args:
            x (torch.Tensor): Tensor de entrada (N, seq_len, embed_size).

        Returns:
            torch.Tensor: Saída processada (N, seq_len, embed_size).
        """
        attention = self.attention(x)
        x = self.norm1(attention + x)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out


class MicroLlama(nn.Module):
    """Arquitetura simplificada inspirada no Llama.

    Combina embeddings de tokens e posições com uma sequência de blocos Transformer
    para predição de tokens em tarefas de modelagem de linguagem.

    Attributes:
        embed_size (int): Dimensão dos vetores de embedding.
        token_embedding (nn.Embedding): Mapeamento de IDs de tokens para vetores.
        position_embedding (nn.Embedding): Mapeamento de posições para vetores.
        layers (nn.ModuleList): Lista contendo os blocos TransformerBlock.
        fc_out (nn.Linear): Camada de saída para o espaço do vocabulário.
    """

    def __init__(
        self,
        vocab_size,
        embed_size=128,
        num_layers=2,
        heads=4,
        forward_expansion=4,
        max_length=100,
    ):
        """Configura a MicroLlama.

        Args:
            vocab_size (int): Tamanho total do vocabulário.
            embed_size (int): Dimensão interna do modelo. Defaults to 128.
            num_layers (int): Quantidade de blocos Transformer. Defaults to 2.
            heads (int): Quantidade de cabeças de atenção. Defaults to 4.
            forward_expansion (int): Expansão do feed forward. Defaults to 4.
            max_length (int): Comprimento máximo da sequência de entrada. Defaults to 100.
        """
        super().__init__()
        self.embed_size = embed_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, forward_expansion)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        """Calcula os logits para a próxima predição de token.

        Args:
            x (torch.Tensor): Tensor de IDs de tokens (N, seq_len).

        Returns:
            torch.Tensor: Logits de saída (N, seq_len, vocab_size).
        """
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(x.device)

        x = self.token_embedding(x) + self.position_embedding(positions)

        for layer in self.layers:
            x = layer(x)

        logits = self.fc_out(x)
        return logits