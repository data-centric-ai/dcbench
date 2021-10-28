import os
from pathlib import Path


LOCAL_DIR = os.path.join(Path.home(), ".dcbench")
PUBLIC_REMOTE_URL = "https://storage.googleapis.com/dcai/dcbench"


ARTEFACTS_DIR = "artefacts"
SCENARIOS_DIR = "scenarios"
SOLUTIONS_DIR = "solutions"
PUBLIC_ARTEFACTS_DIR = "public"
HIDDEN_ARTEFACTS_DIR = "optional"
SUBMITTED_ARTEFACTS_DIR = "submitted"
LOCAL_ARTEFACTS_DIR = "local"

METADATA_FILENAME = "metadata.json"
RESULT_FILENAME = "result.json"

PUBLIC_ARTEFACTS_URL = "https://storage.googleapis.com/dcai/"
HIDDEN_ARTEFACTS_URL = None
