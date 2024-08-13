import os
import time
from typing import List
import json
from pathlib import Path
from ..utils import get_logger


class SamplesFS:
    def __init__(self, checkpoint_dir, samples_file, output_dir, logger, checkpoint_type="json", checkpoint_name=None):
        self._checkpoint_dir = checkpoint_dir
        self._samples_file = samples_file
        self._output_dir = output_dir
        self._logger = logger
        self._checkpoint_name = None

        if checkpoint_type == "json":
            self._checkpoint = JsonCheckpoint(checkpoint_dir, logger)
        else:
            raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")

        self.__reader = SamplesReader(samples_file, logger)

        output_file = Path(output_dir) / f'{time.strftime("%Y%m%d-%H%M%S")}_samples.json'
        self.__writer = SamplesWriter(output_file, logger)

    def load_samples(self, ):
        last_line = self._checkpoint.load(self._checkpoint_name)
        self._logger.info(f"[SamplesFS] Last line: {last_line}")
        return self.__reader.load_samples(last_line)

    def write_samples(self, samples, not_augmented_samples):
        try:
            self.__writer.write_samples(samples)
            self._create_checkpoint(len(samples), not_augmented_samples)
        except Exception as e:
            self._logger.error(f"[SamplesFS] Error writing samples: {e}")
            not_augmented_samples.extend(samples)
            self._create_checkpoint(0, not_augmented_samples)
            self._logger.error(f"Checkpoint created with {len(not_augmented_samples)} not augmented samples")

    def _create_checkpoint(self, size, not_augmented_samples):
        self._checkpoint.create(size, not_augmented_samples)


class Checkpoint:
    def __init__(self, checkpoint_dir, logger):
        self._checkpoint_dir = checkpoint_dir
        self._logger = logger
        self._checkpoint_name = None

    def load(self, checkpoint_name=None) -> List[tuple[str, str, str]]:
        if checkpoint_name is not None:
            return self._load_checkpoint(checkpoint_name)
        else:
            return self._load_last_checkpoint()

    def _load_last_checkpoint(self) -> List[tuple[str, str, str]]:
        last_checkpoint = self._find_last_checkpoint()
        self._checkpoint_name = last_checkpoint
        return self._load_checkpoint(last_checkpoint) if last_checkpoint is not None else -1

    def _find_last_checkpoint(self) -> str:
        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)
            self._logger.info(f"[Checkpoint] Created checkpoint directory: {self._checkpoint_dir}")
        checkpoints = os.listdir(self._checkpoint_dir)
        self._logger.info(f"[Checkpoint] Found {len(checkpoints)} checkpoints")
        return max(checkpoints) if checkpoints else None

    def _load_checkpoint(self, checkpoint_name) -> List[tuple[str, str, str]]:
        raise NotImplementedError

    def create(self, size, not_augmented_samples):
        self._create(size, not_augmented_samples)

    def _create(self, size, not_augmented_samples):
        raise NotImplementedError


class JsonCheckpoint(Checkpoint):
    def __init__(self, checkpoint_dir, logger):
        super().__init__(checkpoint_dir, logger)
        self.last_sample_number = -1

    def _load_checkpoint(self, checkpoint_name) -> List[tuple[str, str, str]]:
        if checkpoint_name is None:
            self._logger.error("[Checkpoint] checkpoint_name is None")
            raise ValueError("checkpoint_name is None")

        file_path = Path(self._checkpoint_dir) / checkpoint_name
        try:
            with open(file_path, "r") as f:
                checkpoint = json.load(f)
                self.last_sample_number = checkpoint["last_sample_number"]
                self._logger.info(f"[Checkpoint] Loaded checkpoint {checkpoint_name}."
                                  f" Last sample number: {self.last_sample_number}")
                return self.last_sample_number

        except FileNotFoundError:
            self._logger.error(f"[Checkpoint] Checkpoint {checkpoint_name} not found")
            raise FileNotFoundError(f"Checkpoint {checkpoint_name} not found")

    def _create(self, size, not_augmented_samples):
        self.last_sample_number += size + len(not_augmented_samples)

        last_checkpoint = self._find_last_checkpoint()
        if last_checkpoint is not None:
            last_checkpoint_path = Path(self._checkpoint_dir) / f"{last_checkpoint}"
            with open(last_checkpoint_path, "r") as f:
                checkpoint = json.load(f)
                not_augmented_samples.extend(checkpoint["not_augmented_samples"])
            os.remove(last_checkpoint_path)
            self._logger.info(f"[JsonCheckpoint] Removed last checkpoint.")

        checkpoint = {
            "last_sample_number": self.last_sample_number,
            "not_augmented_samples": not_augmented_samples
        }
        self._checkpoint_name = f"{time.strftime('%Y%m%d-%H%M%S')}_checkpoint.json"
        with open(Path(self._checkpoint_dir) / f"{self._checkpoint_name}", "w") as f:
            json.dump(checkpoint, f)
        self._logger.info(f"[JsonCheckpoint] Created checkpoint. Last sample number: {self.last_sample_number}")


class SamplesReader:
    def __init__(self, samples_file, logger):
        self._samples_file = samples_file
        self._logger = logger

    def load_samples(self, line_number) -> List[List[str]]:
        try:
            with open(self._samples_file, "r") as f:
                lines = f.readlines()

            if line_number < -1 or line_number >= len(lines):
                raise ValueError("Invalid line number")

            lines_after = lines[line_number + 1:]
            samples = []
            self._logger.info(f"[SamplesReader] Loading {len(lines_after)} samples")
            for line in lines_after:
                samples.append(line.strip().split(";"))

            self._logger.info(f"[SamplesReader] Loaded {len(samples)} samples")
            return samples

        except FileNotFoundError:
            raise FileNotFoundError(f"Samples file not found")


class SamplesWriter:
    def __init__(self, output_file, logger):
        self._output_file = output_file
        self._logger = logger

    def write_samples(self, samples):
        if os.path.exists(self._output_file):
            self.__log_info(f"Appending {len(samples)} samples to {self._output_file}")
            # self._logger.info(f"Appending {len(samples)} samples to {self._output_file}")
            with open(self._output_file, "r") as f:
                output = json.load(f)

            output.extend(samples)
        else:
            self.__log_info(f"Creating new file {self._output_file}")
            output = samples

        with open(self._output_file, "w") as f:
            self.__log_info(f"Writing total {len(output)} samples to {self._output_file}")
            json.dump(output, f)
            self.__log_info(f"Samples written to {self._output_file}")

    def __log_info(self, msg):
        self._logger.info(f"[SamplesWriter] {msg}")


if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))

    checkpoint_dir = Path(path) / ".." / ".." / "res" / "checkpoints"
    log_dir = Path(path) / ".." / ".." / "logs"
    output_dir = Path(path) / ".." / ".." / "out"
    samples_path = Path(path) / ".." / ".." / "res" / "augmentation_relations_subset.csv"

    logger = get_logger(log_dir, "samples_fs", True, False)
    logger.info(f"[MAIN] Checkpoint dir: {checkpoint_dir}")
    logger.info(f"[MAIN] Samples file: {samples_path}")

    fs = SamplesFS(checkpoint_dir, samples_path, output_dir, logger)
    samples = fs.load_samples()

    logger.info(f'[MAIN] Loaded {len(samples)} samples')
    logger.info(f'[MAIN] First sample: {samples[0]}')

    test_samples = [
        {"in": [('a', 'b'), ('c', 'd')], "out": "some output"},
        {"in": [('e', 'f'), ('g', 'h')], "out": "some output"},
        {"in": [('i', 'j'), ('k', 'l')], "out": "some output"}
    ]

    not_augmented_samples = [
        {"in": [('m', 'n'), ('o', 'p')], "out": "some output"},
        {"in": [('q', 'r'), ('s', 't')], "out": "some output"},
        {"in": [('u', 'v'), ('w', 'x')], "out": "some output"}
    ]

    fs.write_samples(test_samples, not_augmented_samples)
    fs.write_samples(test_samples, [])
