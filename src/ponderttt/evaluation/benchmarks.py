"""
Code generation benchmarks (HumanEval, MBPP, ClassEval).
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass


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
        try:
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

        except Exception as e:
            warnings.warn(
                f"Failed to load HumanEval: {e}. Using placeholder.", stacklevel=2
            )
            # Create placeholder problems for testing
            for i in range(5):
                self.problems.append(
                    CodeProblem(
                        task_id=f"HumanEval/{i}",
                        prompt=f"def example_{i}(n):\n    # TODO: Implement\n    ",
                        canonical_solution="    return n * 2\n",
                        test_code=f"assert example_{i}(2) == 4",
                        entry_point=f"example_{i}",
                    )
                )

    def __len__(self) -> int:
        """Number of problems."""
        return len(self.problems)

    def __getitem__(self, idx: int) -> CodeProblem:
        """Get problem by index."""
        return self.problems[idx]

    def evaluate(
        self,
        generate_fn: Callable[[str], list[str]],
        k: int = 100,
    ) -> dict[str, float]:
        """
        Evaluate model on HumanEval.

        Args:
            generate_fn: Function that takes prompt and returns k completions
            k: Number of samples per problem

        Returns:
            Dictionary with pass@k scores
        """
        warnings.warn(
            "Execution-based evaluation requires a safe sandbox. "
            "This is a placeholder implementation.",
            stacklevel=2,
        )

        # Placeholder: return dummy scores
        return {
            "pass@1": 0.0,
            "pass@10": 0.0,
            "pass@100": 0.0,
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
        try:
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

        except Exception as e:
            warnings.warn(f"Failed to load MBPP: {e}. Using placeholder.", stacklevel=2)
            # Create placeholder problems
            for i in range(5):
                self.problems.append(
                    CodeProblem(
                        task_id=f"mbpp/{i}",
                        prompt=f"Write a function to compute {i}",
                        canonical_solution=f"def solution(n):\n    return n + {i}",
                        test_code=f"assert solution(1) == {1 + i}",
                        entry_point="solution",
                    )
                )

    def __len__(self) -> int:
        """Number of problems."""
        return len(self.problems)

    def __getitem__(self, idx: int) -> CodeProblem:
        """Get problem by index."""
        return self.problems[idx]

    def evaluate(
        self,
        generate_fn: Callable[[str], list[str]],
        k: int = 100,
    ) -> dict[str, float]:
        """
        Evaluate model on MBPP.

        Args:
            generate_fn: Function that takes prompt and returns k completions
            k: Number of samples per problem

        Returns:
            Dictionary with pass@k scores
        """
        warnings.warn(
            "Execution-based evaluation requires a safe sandbox. "
            "This is a placeholder implementation.",
            stacklevel=2,
        )

        return {
            "pass@1": 0.0,
            "pass@10": 0.0,
            "pass@100": 0.0,
        }


class ClassEvalBenchmark:
    """
    ClassEval benchmark (100 class-level problems).

    Reference: "ClassEval: A Manually-Crafted Benchmark for Evaluating
                LLMs on Class-level Code Generation"
               Du et al., 2023
    """

    def __init__(self):
        """Initialize ClassEval benchmark."""
        self.problems: list[CodeProblem] = []
        warnings.warn(
            "ClassEval dataset is not yet publicly available on HuggingFace. "
            "Using placeholder.",
            stacklevel=2,
        )
        self._load_problems()

    def _load_problems(self):
        """Load ClassEval problems."""
        # Placeholder: ClassEval not yet on HuggingFace
        for i in range(5):
            self.problems.append(
                CodeProblem(
                    task_id=f"ClassEval/{i}",
                    prompt=f"class Example{i}:\n    def method(self, x):\n        # TODO\n        ",
                    canonical_solution="        return x * 2",
                    test_code=f"assert Example{i}().method(2) == 4",
                    entry_point=f"Example{i}",
                )
            )

    def __len__(self) -> int:
        """Number of problems."""
        return len(self.problems)

    def __getitem__(self, idx: int) -> CodeProblem:
        """Get problem by index."""
        return self.problems[idx]

    def evaluate(
        self,
        generate_fn: Callable[[str], list[str]],
        k: int = 100,
    ) -> dict[str, float]:
        """Evaluate model on ClassEval."""
        warnings.warn("ClassEval evaluation not yet implemented.", stacklevel=2)
        return {"pass@1": 0.0}


class BenchmarkSuite:
    """
    Unified interface for all benchmarks.
    """

    def __init__(
        self,
        include_humaneval: bool = True,
        include_mbpp: bool = True,
        include_classeval: bool = False,
    ):
        """
        Initialize benchmark suite.

        Args:
            include_humaneval: Include HumanEval
            include_mbpp: Include MBPP
            include_classeval: Include ClassEval
        """
        self.benchmarks: dict[
            str, HumanEvalBenchmark | MBPPBenchmark | ClassEvalBenchmark
        ] = {}

        if include_humaneval:
            self.benchmarks["humaneval"] = HumanEvalBenchmark()

        if include_mbpp:
            self.benchmarks["mbpp"] = MBPPBenchmark()

        if include_classeval:
            self.benchmarks["classeval"] = ClassEvalBenchmark()

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
