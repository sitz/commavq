# Max-Score Workflow: Local CPU + Colab GPU

## 1) Local CPU setup (works, but not practical for full run)

```bash
cd /Users/sitesh/Code/commavq
python3.11 -m venv .venv-maxscore
source .venv-maxscore/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt constriction~=0.4.2 "numpy<2"
```

Sanity checks:

```bash
python compression/compress_maxscore.py --help
python compression/decompress_maxscore.py --help
python compression/tune_maxscore.py --help
```

CPU feasibility note:
- This machine benchmarked at about `23s/frame` for the model inner loop.
- Approximate full-run time is not practical on CPU for 5,000 segments.

## 2) Colab GPU workflow (recommended for real compression)

Open Colab with a GPU runtime (T4/L4/A100), then run.
Use a repo copy that includes your `compress_maxscore.py`/`decompress_maxscore.py` changes (your fork/branch, not the upstream repo unless you've pushed your changes):

```bash
!git clone <YOUR_REPO_URL_WITH_MAXSCORE_CHANGES> commavq
%cd commavq
!python -m pip install --upgrade pip
!pip install torch==2.2.2 datasets==4.0.0 tqdm constriction~=0.4.2 "numpy<2"
```

Optional: mount Google Drive for resumable outputs:

```python
from google.colab import drive
drive.mount('/content/drive')
WORK_DIR = "/content/drive/MyDrive/commavq_maxscore_work"
OUT_ZIP = "/content/drive/MyDrive/compression_challenge_submission.zip"
```

Tune on subset (recommended):

```bash
!python compression/tune_maxscore.py \
  --device cuda:0 \
  --num-segments 25 \
  --output-config compression/maxscore_best_config.json \
  --output-report compression/maxscore_tuning_report.json
```

Full compression:

```bash
!python compression/compress_maxscore.py \
  --device cuda:0 \
  --num-segments 5000 \
  --config-json compression/maxscore_best_config.json \
  --work-dir "$WORK_DIR" \
  --submission-zip "$OUT_ZIP" \
  --log-every 10
```

Optional validation in Colab:

```bash
!./compression/evaluate.sh "$OUT_ZIP"
```

## 3) Optional local model path

All maxscore scripts accept `--model-url`.
For a local model file:

```bash
--model-url "file:///absolute/path/to/pytorch_model.bin"
```
