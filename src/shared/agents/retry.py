"""Small retry primitives shared by agent stages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RetryPhase(StrEnum):
    """The small set of lifecycle phases a stage can report."""

    INVOKE = "invoke"
    PARSE = "parse"
    REPAIR = "repair"
    VALIDATE = "validate"
    VALIDATION = "validation"


@dataclass(frozen=True, slots=True)
class RetryContext:
    """Structured information about one failed stage attempt.

    ``error`` intentionally retains the original exception object so callers
    can inspect its type, traceback cause, and any stage-specific attributes.
    """

    attempt: int
    max_attempts: int
    phase: RetryPhase
    error: Exception
    issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.attempt < 1:
            raise ValueError("attempt must be at least 1")
        if self.max_attempts < self.attempt:
            raise ValueError("max_attempts must be greater than or equal to attempt")
        if not isinstance(self.phase, RetryPhase):
            raise TypeError("phase must be a RetryPhase")

    @property
    def attempts_remaining(self) -> int:
        """Number of attempts still available after this failure."""

        return self.max_attempts - self.attempt

    @property
    def is_final_attempt(self) -> bool:
        return self.attempts_remaining == 0

    @property
    def cause(self) -> BaseException | None:
        """The original chained cause, when the stage error has one."""

        return self.error.__cause__

    def describe(self) -> str:
        """Return a compact diagnostic without discarding structured fields."""

        details = f"attempt {self.attempt}/{self.max_attempts} during {self.phase.value}: {self.error}"
        return f"{details}; issues: {'; '.join(self.issues)}" if self.issues else details


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bound retry attempts and decide whether a failed attempt may repeat.

    ``retryable_exceptions_by_phase`` is the precise form for lifecycles whose
    exception contract differs by phase.  When supplied it takes precedence
    over the broader exception/phase filters.
    """

    max_attempts: int = 3
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,)
    retryable_phases: frozenset[RetryPhase] | None = None
    retryable_exceptions_by_phase: tuple[
        tuple[RetryPhase, tuple[type[Exception], ...]], ...
    ] = ()

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int):
            raise TypeError("max_attempts must be an integer")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if not isinstance(self.retryable_exceptions, tuple):
            raise TypeError("retryable_exceptions must be a tuple")
        if not all(isinstance(item, type) and issubclass(item, Exception) for item in self.retryable_exceptions):
            raise TypeError("retryable_exceptions must contain Exception subclasses")
        if self.retryable_phases is not None:
            if not isinstance(self.retryable_phases, frozenset):
                raise TypeError("retryable_phases must be a frozenset")
            if not self.retryable_phases or not all(
                isinstance(phase, RetryPhase) for phase in self.retryable_phases
            ):
                raise TypeError("retryable_phases must contain RetryPhase values")
        if not isinstance(self.retryable_exceptions_by_phase, tuple):
            raise TypeError("retryable_exceptions_by_phase must be a tuple")
        seen_phases: set[RetryPhase] = set()
        for entry in self.retryable_exceptions_by_phase:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError(
                    "retryable_exceptions_by_phase entries must be (RetryPhase, exception tuple) pairs"
                )
            phase, exception_types = entry
            if not isinstance(phase, RetryPhase):
                raise TypeError("retryable_exceptions_by_phase keys must be RetryPhase values")
            if phase in seen_phases:
                raise ValueError("retryable_exceptions_by_phase must not contain duplicate phases")
            seen_phases.add(phase)
            if not isinstance(exception_types, tuple) or not exception_types:
                raise TypeError("phase-specific retryable exceptions must be a non-empty tuple")
            if not all(
                isinstance(item, type) and issubclass(item, Exception)
                for item in exception_types
            ):
                raise TypeError("phase-specific retryable exceptions must contain Exception subclasses")

    def attempts(self) -> range:
        """Yield one-based attempt numbers for this policy."""

        return range(1, self.max_attempts + 1)

    def failure_context(
        self,
        *,
        attempt: int,
        phase: RetryPhase,
        error: Exception,
        issues: tuple[str, ...] = (),
    ) -> RetryContext:
        """Create the structured failure record used for retry decisions."""

        return RetryContext(
            attempt=attempt,
            max_attempts=self.max_attempts,
            phase=phase,
            error=error,
            issues=issues,
        )

    def should_retry(self, context: RetryContext) -> bool:
        """Return whether the supplied failed attempt is eligible to repeat."""

        if context.max_attempts != self.max_attempts or context.is_final_attempt:
            return False
        if self.retryable_exceptions_by_phase:
            for phase, exception_types in self.retryable_exceptions_by_phase:
                if context.phase is phase:
                    return isinstance(context.error, exception_types)
            return False
        if self.retryable_phases is not None and context.phase not in self.retryable_phases:
            return False
        return isinstance(context.error, self.retryable_exceptions)

    def allows_retry(
        self,
        *,
        attempt: int,
        phase: RetryPhase,
        error: Exception,
        issues: tuple[str, ...] = (),
    ) -> bool:
        """Convenience form of :meth:`should_retry` for one failure."""

        return self.should_retry(
            self.failure_context(
                attempt=attempt,
                phase=phase,
                error=error,
                issues=issues,
            )
        )


__all__ = ["RetryContext", "RetryPhase", "RetryPolicy"]
