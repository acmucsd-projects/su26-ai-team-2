# Sign to Sentence: ASL Translator

A webcam-based ASL fingerspelling prototype built by Summer 2026 AI Team 2. It recognizes static letters and digits, combines them into text, and reads the text aloud using ElevenLabs.

## Features

- Live hand tracking and predictions using OpenCV and MediaPipe.
- Sentence building with stable predictions and confidence filtering.
- Custom gestures for adding spaces and triggering speech.
- Streamlit and desktop interfaces.
- SVM and Random Forest classifiers.

## How It Works

MediaPipe detects 21 hand landmarks, which are normalized into 63 coordinate features. The model predicts a character, and the sentence application adds it after 28 matching frames with confidence of at least 0.4. ElevenLabs converts the accumulated text to speech.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install requests playsound python-dotenv
```

Place the provided models in this folder:

```text
models/
├── sign_model.pkl
└── randomForest_model.pkl
```

For speech, create a `.env` file in the project root:

```dotenv
ELEVENLABS_API_KEY=your_api_key_here
```

Keep `.env` out of version control. Recognition works without an API key, but speech requires one.

## Usage

Run all commands from the project root.

### Streamlit App

```bash
streamlit run app.py
```

- Hold a supported sign steady to add a character.
- Spread all five fingers to add a space.
- Bring your thumb near your index and middle fingertips, with your ring and pinky curled, to speak the text.

The current on-screen speech instructions differ from the detector; follow the gesture described above. Run locally to use your webcam.

### Desktop App

```bash
python sentences.py
```

Click the webcam window before using keyboard controls:

| Key | Action |
| --- | --- |
| Space | Add a space |
| Backspace | Delete a character |
| `c` | Clear text |
| `s` | Save to `data/process/sentences.txt` |
| `v` | Speak text |
| `q` or Esc | Quit |

### Random Forest Demo

```bash
python test.py
```

Displays live predictions using `randomForest_model.pkl`. Press `q` to quit.

## Project Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface |
| `sentences.py` | Sentence recognition, desktop controls, and speech |
| `extract.py` | Extract hand landmarks from images |
| `train_model.py` | Train and evaluate the SVM model |
| `modelJL.py` | Train and evaluate the Random Forest model |
| `test.py` | Random Forest webcam demo |

## Training

Organize training images by label under `data/raw/`, such as `data/raw/a/` and `data/raw/b/`.

### Random Forest

```bash
python extract.py
python modelJL.py
```

Extraction creates `data/process/landmarks.csv`. Training saves `models/randomForest_model.pkl`. Skip extraction if the CSV is already available.

### SVM

Before running, update this assignment in `train_model.py` because `extract_dataset()` returns four values:

```python
X, y, groups, is_recovered = extract_dataset()
```

Then run:

```bash
python train_model.py
```

The pipeline uses scaling, rotation augmentation, and grouped evaluation based on `hand<number>_` filename prefixes. At least two distinct hand groups are required. The model is saved to `models/sign_model.pkl`.

## Results and Limitations

The original project README reports **95.89% Random Forest accuracy on 365 test samples**. This is a dataset result, not measured live-app or SVM accuracy.

- **J and Z are excluded** from the sentence application because they require motion.
- The application assembles characters rather than translating full ASL grammar.
- Detection supports one hand at a time.
- Lighting, hand position, and gesture overlap can affect predictions.
- Holding a sign too long can add repeated characters.
- The code requires a MediaPipe installation exposing `mp.solutions.hands`.
