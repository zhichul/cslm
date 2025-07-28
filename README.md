## Codebase for code-switching-language-modeling

Idea is to train a neural language model with vocabulary over two language and through training regularization and/or constrained decoding generate code-switched text, when training only on parallel or separate monoloingual text.

## Structure

Modeling and training code is in `src/cslm/modeling` and `src/cslm/training` respectively. Constrained code-switching sampling (via adapting [stochastic beam search](https://arxiv.org/abs/1903.06059)) are implemented in `src/cslm/inference`.
