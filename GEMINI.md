# GEMINI.md

## Project Overview
**Micro-Llama** is an educational project aimed at mentoring developers in the creation of a minimal Large Language Model (LLM) from scratch. It implements a **Transformer Decoder-only** architecture using Python and PyTorch, focusing on the fundamental concepts of NLP, Deep Learning, and generative AI.

The project is structured to evolve through distinct phases, from basic tokenization to modern LLM features like RoPE and KV Cache.

### Main Technologies
- **Language:** Python
- **Framework:** PyTorch
- **Hardware Acceleration:** MPS (Metal Performance Shaders) for Apple Silicon, falling back to CPU.

---

## Building and Running

### Prerequisites
- Python 3.10+
- PyTorch
- (Optional) Apple Silicon for MPS support

### Training the Model
Para evitar erros de `ModuleNotFoundError`, execute a partir da raiz do projeto configurando o `PYTHONPATH`:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 -m micro_llama.train
```
*Isso irá gerar o arquivo `model.pt` na raiz do projeto.*

### Generating Text
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 -m micro_llama.generate
```
*Métodos de amostragem suportados: Temperature, Top-K, Top-P.*


### Testing
- **TODO:** Implement unit tests for individual Transformer components (Attention, FeedForward).

---

## Development Conventions

### Coding Style
- **Documentation:** Follow Google-style docstrings for functions and classes.
- **Modularity:** Keep the architecture modular (`model.py` for architecture, `tokenizer.py` for data processing).
- **Device Agnostic:** Always check for `mps` or `cuda` before falling back to `cpu`.

### Architecture Roadmap
The project follows a specific maturity classification:
1. **Phase 0 — Fundaments:** Character tokenization and basic data loading.
2. **Phase 1 — Basic Model:** Multi-head Attention and Transformer Blocks.
3. **Phase 2 — Training:** Loss calculation and Backpropagation loop.
4. **Phase 3 — Inference:** Advanced sampling (Temperature, Top-K, Top-P).
5. **Phase 4 — Modern Architecture:** RMSNorm, RoPE, SwiGLU, KV Cache.

### Project Tracking
- **HISTORY.md:** Use this file to document the current phase, completed features, and the next most important step. Update it with every major architectural change.

---

## Project Structure
- `src/micro_llama/`: Core source code.
  - `model.py`: Transformer architecture.
  - `tokenizer.py`: Character-based tokenizer.
  - `train.py`: Training loop and optimization logic.
  - `generate.py`: Inference and sampling logic.
- `data/`: Training corpus (e.g., `data.txt`).
- `docs/`: Additional documentation.
- `HISTORY.md`: Progress and development phase log.
