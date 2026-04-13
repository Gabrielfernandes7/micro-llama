class Tokenizer:
    """Um tokenizador simples baseado em caracteres para codificação de texto.

    Esta classe cria um vocabulário a partir de um texto fornecido, mapeando cada 
    caractere único para um índice inteiro (token) e vice-versa.

    Attributes:
        vocab (dict): Mapeamento de caracteres (str) para índices (int).
        inv_vocab (dict): Mapeamento de índices (int) para caracteres (str).
        vocab_size (int): O número total de caracteres únicos no vocabulário.
    """

    def __init__(self, text):
        """Inicializa o Tokenizer com um vocabulário baseado no texto de entrada.

        Args:
            text (str): O texto base para gerar o vocabulário de caracteres.
        """
        chars = sorted(list(set(text)))
        self.vocab = {ch: i for i, ch in enumerate(chars)}
        self.inv_vocab = {i: ch for ch, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        """Converte uma string em uma lista de tokens inteiros.

        Args:
            text (str): A string a ser codificada.

        Returns:
            list[int]: Uma lista de inteiros representando os caracteres.
        """
        return [self.vocab[c] for c in text]
    
    def decode(self, tokens):
        """Converte uma lista de tokens inteiros de volta para uma string.

        Args:
            tokens (list[int]): Uma lista de índices inteiros.

        Returns:
            str: A representação em texto dos tokens fornecidos.
        """
        return "".join([self.inv_vocab[t] for t in tokens])