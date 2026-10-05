# Spikes

Short, throwaway experiments that inform technology decisions before they are
made in `app/`. Each spike answers one question with evidence (numbers,
outputs, observations) and ends with a recommendation that feeds `DESIGN.md`.

Spike code is not production code: it is not imported by `app/` and is not
part of the test suite.

## Layout

```
spikes/
└── NN-short-name/
    ├── README.md   # question, options compared, method, results, decision
    └── *.py        # runnable experiment(s)
```

Run a spike with the project environment:

```bash
uv run python spikes/NN-short-name/<script>.py
```

## Index

| # | Question | Status | Decision |
|---|----------|--------|----------|
