import torch
import torch.nn as nn
import torch.optim as optim
import os

from micro_llama.model import MicroLlama
from micro_llama.tokenizer import Tokenizer

def train():
    """Executa o pipeline de treinamento da MicroLlama.

    Este processo compreende:
    1. Carregamento dinâmico e tokenização do arquivo 'data.txt' (Machado de Assis).
    2. Identificação e uso de hardware acelerado (Apple Metal Performance Shaders - MPS).
    3. Inicialização do modelo, otimizador (Adam) e função de perda (CrossEntropy).
    4. Loop de treinamento por épocas com janelamento de sequência (Sliding Window).
    5. Cálculo de gradientes e atualização de pesos (Backpropagation).
    6. Persistência dos pesos treinados em um arquivo 'model.pt' na raiz.

    Note:
        O script localiza automaticamente o arquivo em 'src/micro_llama/data/data.txt' 
        independente de onde o processo seja disparado.
        As labels (y) são deslocadas em uma posição à direita em relação aos inputs (x)
        para realizar a tarefa de 'Next Token Prediction'.
    """
    # 1. Carregamento Dinâmico dos Dados
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "data", "data.txt")

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Erro: Arquivo 'data.txt' não encontrado em: {data_path}")
        return

    # 2. Preparação do Tokenizer e Tensores
    tokenizer = Tokenizer(text)
    data = torch.tensor(tokenizer.encode(text))
    vocab_size = tokenizer.vocab_size

    # 3. Configuração de Hardware (Otimizado para MacBook M4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Iniciando treinamento no device: {device}")

    # 4. Hiperparâmetros e Inicialização
    model = MicroLlama(vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    seq_length = 32
    epochs = 10

    # 5. Ciclo de Treinamento
    for epoch in range(epochs):
        model.train()  # Garante que o modelo está em modo de treino
        epoch_loss = 0
        
        for i in range(0, len(data) - seq_length, seq_length):
            # x: tokens atuais | y: alvo (próximo token)
            x = data[i : i + seq_length].unsqueeze(0).to(device)
            y = data[i + 1 : i + seq_length + 1].unsqueeze(0).to(device)

            # Forward pass
            logits = model(x)

            # Reshape para CrossEntropyLoss: [Batch*Seq, Vocab]
            loss = loss_fn(
                logits.view(-1, vocab_size),
                y.view(-1)
            )

            # Otimização (Backpropagation)
            optimizer.zero_grad()   # Reseta gradientes
            loss.backward()          # Calcula erro
            optimizer.step()         # Atualiza pesos

        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    # 6. Persistência do Modelo
    # Salva na raiz do projeto para facilitar a localização pelo generate.py
    save_path = os.path.join(os.getcwd(), "model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Treinamento concluído. Pesos salvos em: {save_path}")

if __name__ == "__main__":
    train()