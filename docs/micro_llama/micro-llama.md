# Micro-Llama: Uma Jornada Matemática pela Arquitetura Transformer 🦙

Este documento serve como um paper técnico e guia de estudos para quem deseja entender a fundo como construir uma Large Language Model (LLM) do zero. O objetivo é que qualquer pessoa possa ler este documento, entender a fundamentação matemática e ser capaz de construir sua própria arquitetura.

---

## 1. Introdução: O Paradigma Autoregressivo

A Micro-Llama é um modelo **Transformer Decoder-only**. O objetivo fundamental é a **Next-Token Prediction** (Predição do Próximo Token).
Matematicamente, queremos modelar a probabilidade condicional:

$$P(x_t | x_{1}, x_{2}, \dots, x_{t-1})$$

Onde $x_t$ é o token no tempo $t$. O modelo aprende a comprimir o conhecimento do mundo (ou de Machado de Assis) em seus pesos para prever a sequência mais provável.

---

## 2. Fundamentação Matemática dos Componentes

### 2.1. Embeddings e Espaço Vetorial
O texto não entra no modelo como letras, mas como vetores. 
$$E = \text{Embedding}(X) \in \mathbb{R}^{L \times d}$$
Onde $L$ é o comprimento da sequência e $d$ é a dimensão do embedding (neste projeto, $d=128$). Isso transforma símbolos discretos em um espaço contínuo onde palavras com significados próximos ficam "perto" uma da outra.

### 2.2. Positional Encoding (Absoluto)
Como o Transformer processa todos os tokens em paralelo (diferente de uma RNN), ele não tem noção inerente de ordem. Somamos um vetor de posição:
$$X = \text{TokenEmbedding} + \text{PositionEmbedding}$$
Isso "carimba" em cada vetor a informação de onde ele está na frase.

### 2.3. O Mecanismo de Auto-Atenção (Self-Attention)
Este é o coração do Transformer. Para cada token, calculamos três vetores usando matrizes de pesos aprendidas $W_Q, W_K, W_V$:
- **Query (Q):** "O que eu estou procurando?" ($Q = XW_Q$)
- **Key (K):** "O que eu tenho a oferecer?" ($K = XW_K$)
- **Value (V):** "A informação que eu carrego." ($V = XW_V$)

A pontuação de atenção é calculada pelo produto escalar escalonado:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- **Por que $\sqrt{d_k}$?** Para evitar que o produto escalar cresça demais e sature o softmax, causando gradientes desaparecendo.

### 2.4. Causal Masking (A Máscara do Futuro)
Para garantir que o modelo não "preveja o presente olhando para o futuro" durante o treino, aplicamos uma máscara triangular inferior:
$$\text{Mask}(S) = \begin{cases} S_{ij} & \text{se } i \ge j \\ -\infty & \text{se } i < j \end{cases}$$
Isso zera a atenção para tokens futuros. No código: `torch.tril(torch.ones(seq_len, seq_len))`.

---

## 3. Inferência Avançada (Fase 3: Sampling)

Após o modelo cuspir os **logits** (pontuações brutas), precisamos escolher o próximo token. Usar apenas o mais provável (Greedy) gera textos repetitivos.

### 3.1. Temperature ($T$)
Escalona os logits antes do softmax:
$$P_i = \frac{\exp(z_i / T)}{\sum \exp(z_j / T)}$$
- $T \to 0$: Torna a distribuição "afiada" (quase Greedy).
- $T > 1$: Torna a distribuição "achatada" (mais criatividade/caos).

### 3.2. Top-K Sampling
Filtra apenas os $K$ tokens mais prováveis. Isso remove a "cauda longa" de tokens irrelevantes que poderiam arruinar a frase.

### 3.3. Top-P (Nucleus) Sampling
Em vez de um número fixo $K$, escolhemos o menor conjunto de tokens cuja soma de probabilidades seja $\ge P$.
$$\sum_{i=1}^V P_i \ge P$$
Isso permite que o modelo seja dinâmico: se ele estiver muito certo, escolhe poucos tokens; se estiver confuso, amplia as opções.

---

## 4. Auditoria e Comparação com LLMs Modernas (Llama 3)

| Recurso | Micro-Llama | Llama 3 / Mistral | Motivação da Mudança |
| :--- | :--- | :--- | :--- |
| **Norm** | LayerNorm | RMSNorm | RMSNorm é computacionalmente mais leve. |
| **Posição** | Absolute | RoPE (Rotary) | RoPE lida melhor com distâncias relativas. |
| **Ativação** | ReLU | SwiGLU | SwiGLU tem melhor expressividade matemática. |

---

## 5. Como Estender este Projeto (Forking)

Se você deseja construir sua própria LLM:
1. **Mude o Tokenizer:** Experimente Byte Pair Encoding (BPE) em vez de caracteres.
2. **Aumente a Profundidade:** Adicione mais camadas em `model.py`.
3. **Mude os Dados:** Treine com código, receitas ou manuais técnicos.
4. **Implemente RoPE:** Substitua o `PositionEmbedding` pela rotação de vetores para suportar contextos de 100k+ tokens.

---

## 🚀 Como usar

Para garantir que as importações funcionem corretamente, execute os comandos a partir da raiz do projeto:

1. **Prepare os dados:** Coloque seu texto em `data/data.txt`.

2. **Configure o ambiente (Linux/macOS):**
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)/src
    ```

3. **Treine o modelo:** 
    ```bash
    python3 -m micro_llama.train
    ```

4. **Gere Texto:**
   ```bash
   python3 -m micro_llama.generate
   ```

---
*Este documento é uma peça viva da evolução do Micro-Llama. Última atualização: Fase 3 concluída.*
