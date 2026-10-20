# LLM Sim Bench

Simulates local LLM hardware and models, estimates tok/s, and runs a speed-matched mock.

- Python 3.11+, standard library only. Keep it dependency-free unless there is a strong reason.
- Code lives in `src/llm_sim_bench/`: `profiles.py` (presets), `estimator.py` (memory/speed
  model), `mock.py` (paced token stream), `server.py` (OpenAI-compatible API), `cli.py`.
- `web/` is a static site with a JavaScript port of the estimator. Python is the reference:
  after changing presets, data or the estimator, run `PYTHONPATH=src python -m llm_sim_bench
  export-web` and keep `web/js/estimator.js` in sync.
- Measurement records live in `data/measurements/` (schema in `data/README.md`); never guess
  missing fields.
- Mock output texts are in `web/data/texts/`: only text written for this project, or
  public-domain works whose author died in or before 1925 and that were first published before
  1931. After changing texts, rerun `tools/measure_tokenizers.py` and `export-web`.
- Run tests: `PYTHONPATH=src python -m unittest discover -s tests -v` and
  `node --test web/test/*.test.mjs`
- All repository text (README, comments, commit messages, issues) is written in English.
  Exception: UI translations in `web/locales/<lang>.json` (`en` is the source; `ko` and `ja`
  are reviewed by the maintainer). Every locale must have the same keys and placeholders.
