# HISTORY

## [2026-06-11] - Fase Atual: Fase 3 — Inferência

### Status do Projeto
O projeto Micro-Llama concluiu os fundamentos (Phase 0), a implementação do modelo básico (Phase 1) e o pipeline de treinamento (Phase 2). Atualmente, o foco está no refinamento do mecanismo de inferência (Phase 3).

### Commits Analisados (Branch: feature/temperature-topk-topp)
- `acbbc79`: add temperature
- `49b0854`: feat: add MicroLLama
- `20cb841`: modify .gitgnore for ignore .pt
- `f2f860e`: add .gitignore
- `fc946a7`: docs: PT - README.md
- `22ffb4a`: Initial commit

### Implementações Realizadas
- **Tokenização:** Tokenizer por caractere (simples e eficaz para aprendizado).
- **Arquitetura:** `MicroLlama` com Multi-head Self-Attention, Masking Causal e Transformer Blocks.
- **Treinamento:** Pipeline completo com otimizador Adam e salvamento de checkpoint.
- **Inferência:** Geração auto-regressiva com suporte a **Temperature**, **Top-K** e **Top-P Sampling**.
- **Documentação:** Criação do paper técnico educacional em `docs/micro_llama/micro-llama.md`.

### Próximo Passo
Transição para a **Fase 4 — Arquitetura Moderna**. Implementação de **RMSNorm** e **RoPE** (Rotary Positional Embeddings) para alinhar o modelo com o estado da arte do Llama original.
