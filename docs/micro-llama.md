# Micro-Llama 🦙

Um modelo de linguagem (LLM) minimalista baseado na arquitetura **Transformer Decoder-only**, implementado em PyTorch. Este projeto é focado em estudos de processamento de linguagem natural (NLP) e treinamento com literatura brasileira.

## 🏗️ Como o algoritmo funciona?

O coração deste projeto é a arquitetura **Transformer**, introduzida no artigo "Attention is All You Need" (2017). Diferente de modelos antigos que liam palavras uma por uma, o Transformer usa um mecanismo chamado **Self-Attention** (Auto-atenção).

### O Ciclo de Vida do Token:
1.  **Tokenização:** O texto de Machado de Assis é quebrado em unidades menores (caracteres).
2.  **Embeddings:** Cada token vira um vetor numérico em um espaço de 128 dimensões.
3.  **Self-Attention:** O modelo calcula a relevância de cada palavra em relação às outras.
    * *Exemplo:* Em "Capitu era oblíqua", o modelo aprende que "Capitu" é o sujeito de "era".
4.  **Next-Token Prediction:** O objetivo final é sempre prever qual a próxima letra baseada no contexto anterior.

## 📁 Estrutura do Projeto
```text
.
├── data/
│   └── data.txt          # Seu texto de Machado de Assis aqui
├── model.pt              # Pesos do modelo após o treino
└── src/
    └── micro_llama/
        ├── model.py      # Definição das camadas neurais
        ├── tokenizer.py  # Conversor Texto <-> Números
        ├── train.py      # Script de treinamento
        └── generate.py   # Script de inferência
```

## 🚀 Como usar

1. **Prepare os dados:** Coloque seu texto em `data/data.txt`.

2. **Treine:** 
    ```bash
    python3 -m micro_llama.train
    ```
3. **Gere Texto:**
   ```bash
   python3 -m micro_llama.generate
   ```

---

### 3. Entendendo as LLMs (Conceitos Chave)

Para seu README e estudos, aqui estão os pilares que fazem sua `MicroLlama` funcionar:

* **Self-Attention (Q, K, V):** Imagine uma biblioteca. A **Query** (Q) é o que você procura, a **Key** (K) é a etiqueta na lombada dos livros, e o **Value** (V) é o conteúdo dentro deles. O modelo cruza essas informações para saber onde focar a "atenção".
    
* **Positional Encoding:** Como o Transformer processa tudo ao mesmo tempo, ele não sabe "quem vem primeiro". Por isso, somamos vetores de posição aos embeddings para que o modelo entenda a ordem das frases de Machado.
* **Causal Masking:** No seu arquivo `model.py`, a linha `torch.tril` cria uma máscara. Isso é o que impede o modelo de "trapacear" no treino: ele só pode ver as palavras passadas para tentar adivinhar a próxima.

