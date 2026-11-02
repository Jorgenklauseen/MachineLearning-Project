## Setup

Requires Python 3.13 and Git. On Windows, use Git Bash.

```bash
# 1. Clone the project
git clone <repo-url>
cd MachineLearning-Project

# 2. Create and activate a virtual environment
py -3.13 -m venv .venv            # Mac/Linux: python3.13 -m venv .venv
source .venv/Scripts/activate     # Mac/Linux: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset (~770 MB, not stored in this repo)
git clone https://github.com/ml4py/dataset-iiit-pet.git data/oxford-iiit-pet
git -C data/oxford-iiit-pet checkout ebb5f9e

# 5. Verify the setup
python src/check_data.py
```

The first line of output should be `3680 3669 37`. Activate the virtual environment again in each new terminal.
