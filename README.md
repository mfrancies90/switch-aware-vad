# switch-aware-vad
# Switch-Aware VAD

**Code-switch-triggered continual learning for voice activity detection on dialectal Arabic speech**

This repository contains the code for the paper *"Switch-Aware Voice Activity Detection: Code-Switch-Triggered Continual Learning for Dialectal Arabic Speech."*

Pretrained voice activity detectors (VADs) are trained mostly on uniform, often monolingual speech. They degrade on dialectal Arabic, where speakers move between Modern Standard Arabic and a regional dialect within a single utterance. This project adapts a compact pretrained MarbleNet VAD to Egyptian code-switched speech with **switch-triggered Elastic Weight Consolidation (EWC)**. The parameter-importance snapshot that protects previously learned behavior is refreshed only when code-switching is detected in the incoming training batch, rather than once at the start or on a fixed schedule.

![Overall framework](figures/fig1_overall_framework.png)

---

## Key results

### DACS-only configuration (unseen validation clips)

| Configuration | Recall on code-switched speech | False-active time (s) |
|---|---|---|
| **Switch-triggered EWC** | **98.3%** | 22.6 |
| Periodic-refresh EWC | 97.8% | 19.6 |
| Vanilla EWC | 92.1% | 55.6 |
| Task-Free MAS | 94.4% | 19.6 |
| Silero VAD (external reference) | n/a | 46.6 |

### Expanded configuration (DACS + 8 Casablanca dialects)

| Configuration | Recall on code-switched speech (retention) | Accuracy on other clips | False-active time (s) |
|---|---|---|---|
| **Switch-triggered EWC** | 98.1% | **99.6%** | **18.0** |
| Periodic-refresh EWC | 99.4% | 99.0% | 36.0 |
| Vanilla EWC | 99.4% | 95.6% | 87.0 |
| Task-Free MAS | 100.0% | 97.5% | 111.0 |

McNemar's exact test confirms that switch-triggered EWC significantly outperforms vanilla EWC at both scales (p = 0.013 and p < 0.001).

![False-active time](figures/Fig2_false_active_time.png)

---

## What the notebook does

The full pipeline is in a single Colab notebook, `notebooks/switch_aware_vad_complete.ipynb`, organized in four phases:

1. **Baseline fine-tuning**: fine-tunes NVIDIA NeMo's pretrained multilingual MarbleNet VAD on DACS speech and MUSAN noise.
2. **Continual-learning comparison**: trains four configurations from the same baseline checkpoint:
   - Switch-triggered EWC (proposed)
   - Periodic-refresh EWC
   - Vanilla EWC (importance estimated once)
   - Task-Free Memory Aware Synapses (loss-plateau refresh)
3. **Explainability**: computes gradient times input saliency maps and mean speech confidence on code-switched speech.
4. **Expanded data, benchmark, and streaming**:
   - Adds speech from 8 Casablanca dialects to the training data.
   - Benchmarks the Silero VAD detector.
   - Runs sliding-window streaming inference (onset latency and fragmentation).

---

## Repository structure

```
switch-aware-vad/
├── notebooks/
│   └── switch_aware_vad_complete.ipynb   # full pipeline, phases 1 to 4
├── figures/                              # figures used in the paper
├── README.md
└── LICENSE
```

---

## Requirements

- Google Colab with a GPU (experiments were run on a single NVIDIA Tesla T4)
- Python 3.10+
- NVIDIA NeMo 3.0.0 with ASR extras, plus `datasets`, `statsmodels`, `librosa`, `soundfile`

```bash
pip install "nemo_toolkit[asr]" datasets statsmodels librosa
```

The notebook installs these itself in its first cells.

---

## Data

All corpora are public. None are redistributed here; please obtain them from their original sources and respect their licenses.

| Corpus | Use | Source |
|---|---|---|
| **DACS** (Egyptian dialectal Arabic code-switching) | Speech and word-level switch annotations | [qcri/Arabic_speech_code_switching](https://github.com/qcri/Arabic_speech_code_switching) |
| **MGB-3** audio | Source audio referenced by DACS | Downloaded automatically from the URL list included in the DACS repository |
| **Casablanca** (8 Arabic dialects) | Multi-dialect training speech and Silero benchmark | [UBC-NLP/Casablanca](https://huggingface.co/datasets/UBC-NLP/Casablanca) on Hugging Face (requires login and accepting the dataset terms) |
| **MUSAN** (noise subset) | Background class | [OpenSLR 17](https://www.openslr.org/17/) (downloaded automatically) |

### Setup steps

1. Download the DACS repository as a ZIP and extract it to:
   ```
   MyDrive/switch-aware-vad/data/raw/dacs_egyptian/Arabic_speech_code_switching-master/
   ```
2. Create a Hugging Face access token and accept the Casablanca dataset terms. The notebook prompts for the token.
3. Open the notebook in Colab and run all cells. It mounts Google Drive and creates all working folders under `MyDrive/switch-aware-vad/`.

Completed phases save checkpoints to Drive, and a rerun loads them instead of retraining. Delete a checkpoint to retrain that stage.

---

## Main hyperparameters

| Setting | Value |
|---|---|
| Baseline fine-tuning | 10 epochs, batch 32, learning rate 1e-4 |
| Continual learning | SGD (momentum 0.9), learning rate 1e-4, batch 16, 5 epochs, gradient clipping 5.0 |
| Penalty weight (EWC and MAS) | 1000 |
| Importance pool | 100 training utterances |
| Switch trigger | at least one code-switched utterance per batch |
| Refresh cooldown | 10 batches (periodic refresh every 11 batches) |
| MAS plateau rule | window of 20 batches, loss range < 0.15 |
| Streaming inference | 0.63 s window, 0.1 s stride, threshold 0.5 |

Data splits use a fixed random seed (`random.seed(0)`).

---

## Citation

If you use this code, please cite:

```bibtex
@article{labib2026switchaware,
  title   = {Switch-Aware Voice Activity Detection: Code-Switch-Triggered Continual Learning for Dialectal Arabic Speech},
  author  = {Labib, Mariam},
  journal = {Under review},
  year    = {2026}
}
```

Please also cite the corpora and tools this work builds on: DACS (Chowdhury et al., Interspeech 2020), MGB-3 (Ali et al., ASRU 2017), Casablanca (Talafha et al., EMNLP 2024), MUSAN (Snyder et al., 2015), MarbleNet (Jia et al., 2020), and NVIDIA NeMo.

---

## License

The code in this repository is released under the MIT License. The datasets and the pretrained MarbleNet checkpoint remain under their original licenses.

## Contact

Mariam Labib: mariamlabib90@gmail.com
