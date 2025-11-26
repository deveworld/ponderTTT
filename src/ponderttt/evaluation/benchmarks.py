"""
Code generation benchmarks (HumanEval, MBPP).
"""

import os
import signal
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable

from tqdm import tqdm

from .metrics import compute_pass_at_k


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


@dataclass
class CodeProblem:
    """A single code generation problem."""

    task_id: str
    prompt: str
    canonical_solution: str
    test_code: str
    entry_point: str


class HumanEvalBenchmark:
    """
    HumanEval benchmark (164 problems).

    Reference: "Evaluating Large Language Models Trained on Code"
               Chen et al., 2021
    """

    def __init__(self):
        """Initialize HumanEval benchmark."""
        self.problems: list[CodeProblem] = []
        self._load_problems()

    def _load_problems(self):
        """Load HumanEval problems."""
        from datasets import load_dataset

        dataset = load_dataset("openai_humaneval", split="test")

        for example in dataset:
            if isinstance(example, dict):
                problem = CodeProblem(
                    task_id=example["task_id"],
                    prompt=example["prompt"],
                    canonical_solution=example["canonical_solution"],
                    test_code=example["test"],
                    entry_point=example["entry_point"],
                )
                self.problems.append(problem)

    def __len__(self) -> int:
        """Number of problems."""
        return len(self.problems)

    def __getitem__(self, idx: int) -> CodeProblem:
        """Get problem by index."""
        return self.problems[idx]

    def evaluate(
        self,
        generate_fn: Callable[[str], Iterable[str]],
        k: int = 100,
    ) -> dict[str, float]:
        scores = []
        attempts_per_problem: list[int] = []

        for problem in tqdm(self.problems, desc="HumanEval"):
            samples = list(generate_fn(problem.prompt))
            if not samples:
                attempts_per_problem.append(0)
                scores.append(0.0)
                continue

            total = min(k, len(samples))
            correct = 0
            for completion in samples[:total]:
                if _check_solution(problem, completion):
                    correct += 1

            attempts_per_problem.append(total)
            scores.append(
                compute_pass_at_k(total, correct, k) if total > 0 else 0.0
            )

        if not scores:
            return {"pass@k": 0.0, "num_problems": 0, "avg_attempts": 0.0}

        return {
            "pass@k": float(sum(scores) / len(scores)),
            "num_problems": len(scores),
            "avg_attempts": float(sum(attempts_per_problem) / len(attempts_per_problem)),
        }


class MBPPBenchmark:
    """
    MBPP benchmark (974 problems).

    Reference: "Program Synthesis with Large Language Models"
               Austin et al., 2021
    """

    def __init__(self):
        """Initialize MBPP benchmark."""
        self.problems: list[CodeProblem] = []
        self._load_problems()

    def _load_problems(self):
        """Load MBPP problems."""
        from datasets import load_dataset

        dataset = load_dataset("mbpp", split="test")

        for example in dataset:
            if isinstance(example, dict):
                problem = CodeProblem(
                    task_id=f"mbpp/{example['task_id']}",
                    prompt=example["text"],
                    canonical_solution=example["code"],
                    test_code="\n".join(example["test_list"]),
                    entry_point="solution",  # MBPP doesn't specify
                )
                self.problems.append(problem)

    def __len__(self) -> int:
        """Number of problems."""
        return len(self.problems)

    def __getitem__(self, idx: int) -> CodeProblem:
        """Get problem by index."""
        return self.problems[idx]

    def evaluate(
        self,
        generate_fn: Callable[[str], Iterable[str]],
        k: int = 100,
    ) -> dict[str, float]:
        scores = []
        attempts_per_problem: list[int] = []

        for problem in tqdm(self.problems, desc="MBPP"):
            samples = list(generate_fn(problem.prompt))
            if not samples:
                attempts_per_problem.append(0)
                scores.append(0.0)
                continue

            total = min(k, len(samples))
            correct = 0
            for completion in samples[:total]:
                if _check_solution(problem, completion):
                    correct += 1

            attempts_per_problem.append(total)
            scores.append(
                compute_pass_at_k(total, correct, k) if total > 0 else 0.0
            )

        if not scores:
            return {"pass@k": 0.0, "num_problems": 0, "avg_attempts": 0.0}

        return {
            "pass@k": float(sum(scores) / len(scores)),
            "num_problems": len(scores),
            "avg_attempts": float(sum(attempts_per_problem) / len(attempts_per_problem)),
        }


class BenchmarkSuite:
    """
    Unified interface for all benchmarks.
    """

    def __init__(
        self,
        include_humaneval: bool = True,
        include_mbpp: bool = True,
    ):
        """
        Initialize benchmark suite.

        Args:
            include_humaneval: Include HumanEval
            include_mbpp: Include MBPP
        """
        self.benchmarks: dict[
            str, HumanEvalBenchmark | MBPPBenchmark
        ] = {}

        if include_humaneval:
            self.benchmarks["humaneval"] = HumanEvalBenchmark()

        if include_mbpp:
            self.benchmarks["mbpp"] = MBPPBenchmark()

    def evaluate_all(
        self,
        generate_fn: Callable[[str], list[str]],
        k: int = 100,
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate on all benchmarks.

        Args:
            generate_fn: Generation function
            k: Number of samples

        Returns:
            Dictionary mapping benchmark name to results
        """
        results = {}

        for name, benchmark in self.benchmarks.items():
            print(f"Evaluating on {name}...")
            results[name] = benchmark.evaluate(generate_fn, k=k)

        return results

    def get_benchmark(self, name: str):
        """Get specific benchmark by name."""
        return self.benchmarks.get(name)


def _check_solution(problem: CodeProblem, completion: str, timeout: int = 3) -> bool:
    """Execute completion+tests to determine correctness."""
    if os.environ.get("PONDER_TTT_ALLOW_UNSAFE_BENCHMARKS") != "1":
        raise RuntimeError(
            "Unsafe code execution is disabled. Set PONDER_TTT_ALLOW_UNSAFE_BENCHMARKS=1 "
            "to run benchmark tests in a trusted, sandboxed environment."
        )

    namespace: dict[str, object] = {}
    source = f"{problem.prompt}\n{completion}\n"

    try:
        with time_limit(timeout):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=SyntaxWarning)
                exec(compile(source, "<completion>", "exec"), namespace, namespace)  # noqa: S102
                exec(compile(problem.test_code, "<tests>", "exec"), namespace, namespace)  # noqa: S102
        return True
    except Exception:
        return False