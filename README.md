# Virtual Gym Assistant (CV)

Computer-vision project with two exercise rep-counting approaches:

- **Heuristic approach** (angle/rule-based): `heuristic/app.py`
- **Phase approach** (model-based phase detection): `phase/app.py`

## Dataset
We used the Penn Action dataset: https://dreamdragon.github.io/PennAction/
We augmented this dataset with the algorithms found in the augmentation folder. First we ran exercise_selection.py to get all the squat and pushup files 
and then we used the augmentation.py to create basic augmentations and the loop.py to create loops in the examples to simulate more reps per example. 

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
python heuristic/app.py
```

### 2) Phase approach
Rep counting through a trained/model phase pipeline (instead of angle thresholds):

```bash
python phase/app.py
```

## Project structure (overview)

```text
virtual-gym-assistant-CV/
├─ ReadMe.md
├─ requirements.txt
├─ heuristic/
│  └─ app.py
└─ phase/
    └─ app.py
```

## Notes

- Use **Python 3.10** to avoid dependency/version issues.
- Run commands from the project root directory (where `requirements.txt` exists).