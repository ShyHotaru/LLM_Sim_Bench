# LLM Sim Bench

Simulates local LLM hardware and models, estimates tok/s, and runs a speed-matched mock.

- Python 3.11+, standard library only. Keep it dependency-free unless there is a strong reason.
- Code lives in `src/llm_sim_bench/`: `profiles.py` (presets), `estimator.py` (memory/speed
  model), `mock.py` (paced token stream), `server.py` (OpenAI-compatible API), `cli.py`.
- Run tests: `PYTHONPATH=src python -m unittest discover -s tests -v`
- All repository text (README, comments, commit messages, issues) is written in English.
