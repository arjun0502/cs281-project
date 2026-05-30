import os
from datetime import datetime

from prompt_opt_pipeline import config

_LOG_PATH = os.path.join(config.OUTPUTS_DIR, "run_log.txt")


def setup_logging() -> None:
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    with open(_LOG_PATH, "a") as f:
        f.write(
            f"\n{'='*60}\nRun started: {datetime.now().isoformat()}\n{'='*60}\n"
        )


def log(message: str) -> None:
    print(message)
    with open(_LOG_PATH, "a") as f:
        f.write(message + "\n")


def save_prompt(prompt: str, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(prompt)
