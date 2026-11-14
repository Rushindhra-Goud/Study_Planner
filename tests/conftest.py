import json
import shutil
from pathlib import Path

import pytest

from study_planner.config import load_settings
from study_planner.generators import LLMGenerator
from study_planner.ingest import ingest_directory
from study_planner.vectorstore import VectorStore

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "course_materials"


@pytest.fixture(scope="session")
def ingest_result():
    return ingest_directory(DATA, 900)


@pytest.fixture(scope="session")
def store(ingest_result):
    return VectorStore.build(ingest_result.chunks)


@pytest.fixture()
def settings(tmp_path):
    return load_settings(output_dir=tmp_path / "out", store_dir=tmp_path / "store",
                         llm_provider="offline", langsmith_api_key=None)


class FakeLLM:
    """Returns scripted replies, one per call (the last reply repeats)."""

    provider = "fake"
    model = "fake-1"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def complete(self, system, user, max_tokens=2048):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


class FakeLangSmithClient:
    def __init__(self, fail=False):
        self.fail = fail
        self.feedback = []
        self.datasets = {}
        self.examples = []

    def create_feedback(self, run_id, key, **kwargs):
        if self.fail:
            raise ConnectionError("network down")
        self.feedback.append({"run_id": run_id, "key": key, **kwargs})

    def has_dataset(self, dataset_name):
        return dataset_name in self.datasets

    def create_dataset(self, name, description=None):
        class D:
            id = "dataset-1"
        self.datasets[name] = D()
        return self.datasets[name]

    def read_dataset(self, dataset_name):
        return self.datasets[dataset_name]

    def create_example(self, **kwargs):
        if self.fail:
            raise ConnectionError("network down")
        self.examples.append(kwargs)


def make_generator(replies):
    return LLMGenerator(FakeLLM(replies))
