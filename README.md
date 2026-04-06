# Virtual Gym Assistant (CV)

Computer-vision project with two exercise rep-counting approaches:

- **Heuristic approach** (angle/rule-based): `heuristic/app.py`
- **Phase approach** (model-based phase detection): `phase/app.py`
- **Blog Post**: https://rwsteen.github.io/gym-exercises-CV/

## Dataset
We used the Penn Action dataset: https://dreamdragon.github.io/PennAction/.
We augmented this dataset with the algorithms found in the ```augmentation folder```. First we ran ```exercise_selection.py``` to get all the squat and pushup files 
and then we used the ```augmentation.py``` to create basic augmentations and the ```loop.py``` to create loops and simulate more reps per example. We used this augmented dataset to train our models.

## Prerequisites

- **Python 3.10**
- `pip` available in terminal

## Setup

```bash
# (optional) create and activate a virtual environment
python3.10 -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Run the project

### 1) Heuristic approach
Rule/angle-based rep counting:

```bash
streamlit run heuristic/app.py
```

### 2) Phase approach
Rep counting through a trained/model phase pipeline (instead of angle thresholds):

```bash
streamlit run phase/app.py
```

## Project structure (overview)

```text
virtual-gym-assistant-CV/
├─ augmentation/        # Dataset preparation and augmentation scripts
│  ├─ exercise_selection.py
│  ├─ augmentation.py
│  └─ loop.py
├─ docs/                # Project blog
├─ form.py              # Form-analysis utilities (shared between approaches)
├─ heuristic/           # Rule-based rep counting and form analysis pipeline
│  ├─ app.py            # Main heuristic demo entry point
│  ├─ model.py
│  ├─ preprocess.py
│  └─ train.py
├─ model_utils/         # Shared graph and temporal network building blocks
│  ├─ gcn.py
│  ├─ graph.py
│  └─ tcn.py
├─ phase/               # Learning-based phase prediction pipeline
│  ├─ app.py            # Main phase-based demo entry point
│  ├─ model.py
│  ├─ preprocess.py
│  └─ train.py
├─ README.md
└─ requirements.txt
```

The two main runnable entry points are `heuristic/app.py` and `phase/app.py`.
The `augmentation/augmented_penn/` folder contains the augmented dataset and can become large, so it is treated as derived data rather than source code.

## Notes

- Use **Python 3.10** to avoid dependency/version issues.
- Run commands from the project root directory (where `requirements.txt` exists).
