# PonderTTT Documentation Index

**Quick Navigation Guide**

---

## 📖 Start Here

### First Time?
1. **[README.md](README.md)** - Project overview and quick start
2. **[STATUS.md](STATUS.md)** - Current status and progress
3. **[QUICKSTART.md](QUICKSTART.md)** - Run experiments immediately

### Want Details?
4. **[PLAN.md](PLAN.md)** - Full 2-month timeline and roadmap
5. **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical implementation details
6. **[PROGRESS_UPDATE.md](PROGRESS_UPDATE.md)** - Comprehensive progress report

---

## 🎯 By Goal

### I want to understand the project
→ [README.md](README.md) - Abstract, problem, solution, results

### I want to run experiments
→ [QUICKSTART.md](QUICKSTART.md) - Commands and configurations

### I want to know what's done
→ [STATUS.md](STATUS.md) - Current status and metrics

### I want to see the timeline
→ [PLAN.md](PLAN.md) - Week-by-week plan to arXiv

### I want technical details
→ [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Architecture and code

### I want a full update
→ [PROGRESS_UPDATE.md](PROGRESS_UPDATE.md) - Everything accomplished

---

## 📁 By Component

### Data
- **Code**: `src/ponderttt/data/wikitext.py`
- **Docs**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md#1-data-pipeline)

### Models
- **Code**: `src/ponderttt/models/transformer_ttt.py`
- **Docs**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md#2-transformer-model-with-ttt)

### Experiments
- **Code**: `experiments/wikitext2_experiment.py`
- **Docs**: [experiments/README.md](experiments/README.md)
- **Guide**: [QUICKSTART.md](QUICKSTART.md#run-experiments)

### Analysis
- **Code**: `experiments/analyze_wikitext2.py`
- **Docs**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md#4-analysis--visualization)

---

## 📊 Results & Figures

### Phase 1 Results
- **Summary**: [README.md](README.md#results) - Phase 1 section
- **Figures**: `experiments/figures/` (correlation_scatter.png, etc.)

### WikiText-2 Results
- **Summary**: [README.md](README.md#results) - WikiText-2 section
- **Figures**: `experiments/figures/pareto_curve_wikitext2.png`

### Demo Visualizations
- **Pareto Curve**: `experiments/figures/pareto_curve_wikitext2.png`
- **Allocation**: `experiments/figures/allocation_distribution.png`
- **Training**: `experiments/figures/training_curves.png`

---

## 🔍 Quick Answers

### How do I run experiments?
See [QUICKSTART.md](QUICKSTART.md) sections:
- Quick Test (1-2 minutes)
- Run Experiments (full instructions)

### What are the results?
See [STATUS.md](STATUS.md#-demo-results-achieved)

### What's been implemented?
See [PROGRESS_UPDATE.md](PROGRESS_UPDATE.md#-objectives-completed)

### What's the architecture?
See [PROGRESS_UPDATE.md](PROGRESS_UPDATE.md#-architecture-implemented)

### How do I contribute?
See [PLAN.md](PLAN.md) for upcoming tasks

### Where are the tests?
See `experiments/test_setup.py` and [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md#5-testing--validation)

---

## 📚 Documentation Map

```
ponderttt/
├── README.md                       ← Project overview
├── PLAN.md                         ← 2-month roadmap
├── STATUS.md                       ← Current progress
├── QUICKSTART.md                   ← How to run
├── IMPLEMENTATION_SUMMARY.md       ← Technical details
├── PROGRESS_UPDATE.md              ← Comprehensive update
├── INDEX.md                        ← This file
│
├── experiments/
│   ├── README.md                   ← Experiment guide
│   ├── wikitext2_experiment.py     ← Main experiments
│   ├── analyze_wikitext2.py        ← Analysis tools
│   ├── test_setup.py               ← Tests
│   ├── quick_demo.py               ← Quick validation
│   ├── mini_experiment.py          ← Mini training
│   └── generate_demo_results.py    ← Demo data
│
└── src/ponderttt/
    ├── data/wikitext.py            ← Dataset
    ├── models/transformer_ttt.py   ← Model
    ├── models/ttt_linear.py        ← TTT layer
    └── models/adaptive_ttt.py      ← Adaptive wrapper
```

---

## 🎓 Learning Path

### Beginner
1. [README.md](README.md) - Understand the problem
2. [QUICKSTART.md](QUICKSTART.md) - Run quick test
3. [STATUS.md](STATUS.md) - See what works

### Intermediate
4. [experiments/README.md](experiments/README.md) - Experiment details
5. [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - How it works
6. Run `experiments/quick_demo.py` - See it in action

### Advanced
7. [PROGRESS_UPDATE.md](PROGRESS_UPDATE.md) - Deep dive
8. [PLAN.md](PLAN.md) - Research roadmap
9. Read source code in `src/ponderttt/`

---

## 🚀 Common Tasks

### Run Tests
```bash
uv run python experiments/test_setup.py
```
**Docs**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md#5-testing--validation)

### Generate Demo Results
```bash
uv run python experiments/generate_demo_results.py
uv run python experiments/analyze_wikitext2.py
```
**Docs**: [QUICKSTART.md](QUICKSTART.md#run-experiments)

### Full Experiments
```bash
uv run python experiments/wikitext2_experiment.py --mode all
```
**Docs**: [experiments/README.md](experiments/README.md)

### Check Status
```bash
cat STATUS.md
```
**Docs**: [STATUS.md](STATUS.md)

---

## 📞 Quick Reference

| Need | Document |
|------|----------|
| **What is PonderTTT?** | [README.md](README.md) |
| **How do I use it?** | [QUICKSTART.md](QUICKSTART.md) |
| **What's the status?** | [STATUS.md](STATUS.md) |
| **What's the plan?** | [PLAN.md](PLAN.md) |
| **How does it work?** | [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) |
| **What's been done?** | [PROGRESS_UPDATE.md](PROGRESS_UPDATE.md) |
| **How to run experiments?** | [experiments/README.md](experiments/README.md) |

---

## 📈 Version History

- **v0.1.0** (Nov 2-4, 2025) - Phase 1: Synthetic experiments
- **v0.2.0** (Nov 5, 2025) - Days 4-5: WikiText-2 implementation ✅
- **v0.3.0** (TBD) - Days 6-7: Analysis & documentation
- **v1.0.0** (TBD) - arXiv submission

---

## 🔗 External Resources

### Papers Referenced
- TTT-Linear (original)
- LaCT (arXiv:2505.23884)
- Titans (arXiv:2501.00663)
- MGG (arXiv:2412.16901)

### Datasets
- WikiText-2 (HuggingFace)
- WikiText-103 (planned)
- Penn Treebank (planned)

---

*PonderTTT Documentation Index*
*Last Updated: November 5, 2025*
