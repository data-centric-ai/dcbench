import os
from pathlib import Path

LOCAL_DIR = os.path.join(Path.home(), ".dcbench")
BUCKET_NAME = "dcbench"
HIDDEN_BUCKET_NAME = "dcbench-hidden"
PUBLIC_REMOTE_URL = f"https://storage.googleapis.com/{BUCKET_NAME}"


ARTEFACTS_DIR = "artefacts"
PROBLEMS_DIR = "problems"
SOLUTIONS_DIR = "solutions"
PUBLIC_ARTEFACTS_DIR = "public"
HIDDEN_ARTEFACTS_DIR = "optional"
LOCAL_ARTEFACTS_DIR = "local"

METADATA_FILENAME = "metadata.json"
RESULT_FILENAME = "result.json"

PUBLIC_ARTEFACTS_URL = f"https://storage.googleapis.com/{BUCKET_NAME}/"
HIDDEN_ARTEFACTS_URL = None
