# Base Imports
import logging
import subprocess
import time
from pathlib import Path

# Third party imports
import streamlit as st


def get_logger(name: str, logfile: str, level=logging.INFO):
    """Return a logger that overwrites the log file each run."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # <- important

    # Remove existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(logfile, mode="w")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(file_handler)

    return logger


class TqdmLogger:
    def __init__(self, logger, log_every=10):
        self.logger = logger
        self.log_every = log_every
        self._counter = 0

    def write(self, msg):
        msg = msg.strip()
        if msg:
            self._counter += 1
            if self._counter % self.log_every == 0:
                self.logger.info(msg)

    def flush(self):
        pass


def stream_log_until_done(
    process: subprocess.Popen,
    log_file: str,
    refresh_interval: float = 0.5,
    language: str = "bash",
):
    """
    Stream a log file live in Streamlit until the given subprocess finishes.

    Parameters
    ----------
    process : subprocess.Popen
        The process writing to the log file.
    log_file : str
        Path to the log file.
    refresh_interval : float
        Seconds between reading new lines.
    language : str
        Syntax highlighting in Streamlit.
    """
    log_path = Path(log_file)
    st.write(f"### 📜 Streaming `{log_file}` ...")
    placeholder = st.empty()
    last_pos = 0
    buffer = ""

    while process.poll() is None:  # while process is running
        if log_path.exists():
            with open(log_path, "r") as f:
                f.seek(last_pos)
                new_data = f.read()
                if new_data:
                    buffer += new_data
                    placeholder.code(buffer, language=language)
                last_pos = f.tell()
        time.sleep(refresh_interval)

    # Final read after process ends
    if log_path.exists():
        with open(log_path, "r") as f:
            buffer = f.read()
            placeholder.code(buffer, language=language)

    st.success("✅ Process finished.")
