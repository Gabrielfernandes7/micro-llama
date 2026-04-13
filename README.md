# Micro-Llama 🦙

Uma implementação educacional de uma LLM (Large Language Model) mínima, construída do zero em PyTorch, baseada na arquitetura **Transformer Decoder-only**.

> Objetivo: ensinar como uma LLM funciona internamente — da tokenização à geração de texto.

---

## 📌 Visão Geral

O `Micro-Llama` é um projeto didático focado em explicar, de forma simples e prática, como modelos generativos de texto (como GPT) funcionam.

O modelo é treinado **do zero (training from scratch)** utilizando um corpus textual (ex: obras de Machado de Assis).

Você irá aprender:

- Como texto vira números (tokenização)
- Como o modelo aprende padrões (treinamento)
- Como funciona o mecanismo de atenção (self-attention)
- Como o modelo gera texto (inferência)

---

## Arquitetura

O modelo segue o padrão **autoregressivo (decoder-only)**:

```

Texto → Tokenização → Embeddings → Transformer → Linear → Softmax → Próximo token

```

O coração do modelo é o **Transformer**, que usa o mecanismo de **Self-Attention** para entender as relações entre palavras.

## Como o algoritmo funciona

### Ciclo de vida do token

1. **Tokenização**
   - Texto é convertido em unidades menores (caracteres)
   - Exemplo: `"gato"` → `['g','a','t','o']`

2. **Embeddings**
   - Cada token vira um vetor numérico (ex: 128 dimensões)

3. **Positional Encoding**
   - Adiciona informação de ordem (posição da palavra na frase)

4. **Self-Attention (Q, K, V)**
   - O modelo aprende relações entre tokens

   Analogia:
   - Query (Q): o que estou procurando
   - Key (K): onde procurar
   - Value (V): informação encontrada

5. **Causal Masking**
   - Impede o modelo de ver o futuro
   - Implementado com `torch.tril`

6. **Next-Token Prediction**
   - O modelo aprende a prever o próximo token

---

## Exemplo de Treinamento

Entrada:
```

o gato subiu no

```

Alvo:
```

gato subiu no telhado

```

O modelo aprende a prever:

```

telhado

```

---

## Geração de Texto

Após o treino:

1. Recebe um prompt
2. Prediz o próximo token
3. Adiciona ao contexto
4. Repete

Estratégias:
- Greedy
- Temperature
- Top-k / Top-p

---

## Estrutura do Projeto

```

micro-llama/
├── data/
│   └── data.txt
├── src/
│   └── micro_llama/
│       ├── tokenizer.py
│       ├── model.py
│       ├── train.py
│       ├── generate.py
│       └── config.py
├── tests/
├── docs/
├── notebooks/
└── README.md

```

---

## Como usar

### 1. Preparar os dados

Coloque seu texto em:

```

data/data.txt

```

Sugestão: textos de domínio específico (ex: literatura brasileira)

---

### 2. Treinar o modelo

```

python3 -m train.py

```

Isso irá gerar:

```

model.pt

```

---

### 3. Gerar texto

```

python3 -m generate.py

```

---

## Roadmap de Aprendizado

### Fase 1 — Fundamentos
- [ ] Tokenização (caractere)
- [ ] Embeddings

### Fase 2 — Núcleo
- [ ] Self-Attention (Q, K, V)
- [ ] Transformer Block

### Fase 3 — Treinamento
- [ ] Pipeline de treino
- [ ] Loss (Cross-Entropy)
- [ ] Otimizador (Adam)

### Fase 4 — Inferência
- [ ] Geração de texto
- [ ] Sampling

### Fase 5 — Evolução
- [ ] Multi-head attention
- [ ] Tokenização BPE
- [ ] Dataset maior

---

## Limitações

Este projeto é educacional:

- Modelo pequeno
- Treinado com poucos dados
- Baixa generalização
- Não representa LLMs comerciais

---

## Referências

- Attention Is All You Need — https://arxiv.org/abs/1706.03762

- Language Models are Few-Shot Learners — https://arxiv.org/abs/2005.14165

- GPT-4 Technical Report — https://arxiv.org/abs/2303.08774

---

## Objetivo final

Ao concluir este projeto, você será capaz de:

- Entender como uma LLM funciona internamente
- Implementar uma versão simplificada do zero
- Explicar os principais conceitos de modelos modernos

---

## Contribuição

Projeto aberto para aprendizado coletivo.

---

## Autor

Gabriel Fernandes

