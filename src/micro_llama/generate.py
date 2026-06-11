import torch
import torch.nn.functional as F
import os

from micro_llama.model import MicroLlama
from micro_llama.tokenizer import Tokenizer

def generate(
        prompt, 
        max_new_tokens=100,
        temperature=1.0,
        top_k=None,
        top_p=None
    ):
    """Gera texto a partir de um prompt inicial usando o modelo treinado.

    Este processo utiliza a estratégia de amostragem (sampling) para prever 
    iterativamente os próximos tokens. O modelo é colocado em modo de avaliação 
    para desabilitar comportamentos de treino como o cálculo de gradientes.

    Args:
        prompt (str): O texto inicial para começar a geração.
        max_new_tokens (int): A quantidade máxima de novos caracteres a serem 
            gerados. Defaults to 100.
        temperature (float): Controla a aleatoriedade (mais alto = mais criativo).
        top_k (int, optional): Filtra apenas os K tokens mais prováveis.
        top_p (float, optional): Nucleus sampling - filtra tokens que somam a prob P.

    Returns:
        str: O texto original concatenado com o conteúdo gerado pelo modelo.

    Note:
        O script localiza dinamicamente o 'data.txt' e o 'model.pt' baseando-se 
        na estrutura de pastas do projeto. Utiliza aceleração MPS (Metal) se 
        disponível no hardware Apple Silicon.
    """

    # 1. Localização dinâmica de arquivos
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "data", "data.txt")
    
    # O model.pt foi salvo na raiz pelo train.py
    project_root = os.path.dirname(os.path.dirname(base_dir))
    model_path = os.path.join(project_root, "model.pt")

    # 2. Reconstrução do Vocabulário
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return f"Erro: data.txt não encontrado em {data_path}"

    tokenizer = Tokenizer(text)
    vocab_size = tokenizer.vocab_size

    # 3. Configuração de Hardware (M4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 4. Inicialização e Carga do Modelo
    model = MicroLlama(vocab_size).to(device)
    try:
        # map_location garante que o modelo carregue no dispositivo correto
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        return f"Erro: model.pt não encontrado em {model_path}. Treine o modelo primeiro."
    
    model.eval()  # Modo de inferência

    tokens = tokenizer.encode(prompt)

    # 5. Loop Auto-regressivo de Geração
    for _ in range(max_new_tokens):
        # Janela de contexto (últimos 32 tokens) enviados para o device
        x = torch.tensor(tokens[-32:]).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)

        # Seleção do último token (Next Token Prediction)
        next_token_logits = logits[0, -1]

        # Aplicar temperatura
        next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.0)

        # Top-K Sampling
        if top_k is not None:
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[..., [-1]]] = float('-inf')

        # Top-P (Nucleus) Sampling
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens com probabilidade cumulativa acima do threshold (top_p)
            sorted_indices_to_remove = cumulative_probs > top_p
            # Mantém pelo menos o primeiro token (mais provável)
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')

        probs = F.softmax(next_token_logits, dim=0)

        # Sampling multinomial para evitar repetições mecânicas
        next_token = torch.multinomial(probs, num_samples=1).item()
        tokens.append(next_token)

    return tokenizer.decode(tokens)


if __name__ == "__main__":
    # Exemplo de uso com tema Machadiano, Top-K e Top-P
    print("\n Gerando continuação no estilo de Machado de Assis (Advanced Sampling) \n")
    resultado = generate(
        "Capitu, apesar de tudo, é ", 
        max_new_tokens=150, 
        temperature=0.8, 
        top_k=40, 
        top_p=0.9
    )
    print(resultado)