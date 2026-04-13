# Micro Llama

Algoritmo de modelo generativo de texto ollama, repositório para fins educativos. Modelo generativo de texto.

Uma implementação educacional de uma LLM (Large Language Model) mínima, construída do zero com foco em aprendizado prático e entendimento do algoritmo por trás de modelos como GPT.

Segue um `README.md` inicial, estruturado para ensino progressivo e publicação no GitHub:


> Objetivo: ensinar como uma LLM funciona internamente — da tokenização à geração de texto.

---

## 📌 Visão Geral

O `Micro Llama` é um projeto didático que implementa uma LLM **decoder-only** baseada na arquitetura Transformer.

A proposta é responder à pergunta:

> Como uma LLM realmente funciona por dentro?

Você irá entender:

- Como texto vira números (tokenização)
- Como o modelo aprende padrões (treinamento)
- Como ele gera texto (inferência)
- Como funciona o mecanismo de atenção (self-attention)

---

## 🧠 Arquitetura

O modelo segue o padrão de modelos autoregressivos (estilo GPT):

Texto → Tokenização → Embeddings → Transformer Blocks → Linear → Softmax → Próximo token

### Componentes principais:

1. **Tokenizador**
   - Converte texto em IDs numéricos
   - Estratégias: BPE, WordPiece ou simples (para estudo)

2. **Embeddings**
   - Representação vetorial dos tokens
   - Soma com embeddings posicionais

3. **Self-Attention (Masked)**
   - Permite que o modelo "olhe" para tokens anteriores
   - Usa máscara causal (não vê o futuro)

4. **Feed Forward (MLP)**
   - Camada densa aplicada após atenção

5. **Normalização + Residual**
   - Estabiliza o treinamento

6. **Head de saída**
   - Converte embeddings em probabilidades sobre o vocabulário

---

## ⚙️ Como o modelo aprende

O treinamento segue o paradigma:

> Prever o próximo token dado um contexto

Exemplo:

Entrada:  "o gato subiu no"
Alvo:     "telhado"

Função de perda:
- Cross-Entropy Loss

Otimização:
- Adam

---

## 🔁 Inferência (Geração de texto)

Após treinado, o modelo gera texto assim:

1. Recebe um prompt inicial
2. Prediz o próximo token
3. Adiciona ao contexto
4. Repete o processo

Estratégias de geração:
- Greedy
- Temperature
- Top-k / Top-p

---

## 🗂️ Estrutura do Projeto

```
micro-llama/
├── docs/
├── notebooks/
├── src/
│   └── micro_llama/
│       ├── tokenizer.py
│       ├── model.py
│       ├── train.py
│       ├── generate.py
│       └── config.py
├── tests/
└── README.md
```

---

## 🚀 Roadmap de Aprendizado

### Fase 1 — Fundamentos
- [ ] Tokenização do zero
- [ ] Entender embeddings

### Fase 2 — Núcleo da LLM
- [ ] Implementar self-attention
- [ ] Construir um Transformer block

### Fase 3 — Treinamento
- [ ] Pipeline de treino
- [ ] Dataset simples

### Fase 4 — Geração
- [ ] Gerar texto
- [ ] Ajustar sampling

### Fase 5 — Melhorias
- [ ] Multi-head attention
- [ ] Positional encoding avançado
- [ ] Dataset maior

---

## 📊 Limitações

Este projeto é **educacional**:

- Modelo pequeno
- Baixa capacidade de generalização
- Não substitui LLMs reais

---

## 📚 Referências

- Vaswani et al. (2017) — *Attention Is All You Need*
- Brown et al. (2020) — *Language Models are Few-Shot Learners*
- OpenAI (2023) — *GPT-4 Technical Report*
- Hugging Face — Tokenizers e Transformers

---

## 🎯 Objetivo final

Ao concluir este projeto, você será capaz de:

- Explicar como uma LLM funciona internamente
- Implementar uma versão simplificada do zero
- Entender os principais componentes de modelos modernos

## 🤝 Contribuição

Este projeto é aberto para aprendizado coletivo. Sugestões e melhorias são bem-vindas.

## 📌 Autor

Gabriel Fernandes

Se quiser, o próximo passo é gerar automaticamente:

* `tokenizer.py` (do zero)
* `model.py` com self-attention implementado
* primeiro notebook explicativo

Isso já te coloca com um repositório funcional em poucas etapas.
