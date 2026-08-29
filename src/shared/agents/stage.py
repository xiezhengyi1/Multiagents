"""A narrow runner for one agent stage and its bounded repair attempts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterable, TypeVar

from .retry import RetryContext, RetryPhase, RetryPolicy


StageValue = TypeVar("StageValue")


@dataclass(frozen=True, slots=True)
class StageAttempt:
    """One-based attempt metadata supplied to stage callbacks."""

    attempt: int
    max_attempts: int

    @property
    def is_final_attempt(self) -> bool:
        return self.attempt >= self.max_attempts


class StageValidationError(RuntimeError):
    """Default terminal error for a validator that returns issue strings."""

    def __init__(self, issues: tuple[str, ...]) -> None:
        self.issues = issues
        super().__init__("Stage validation failed: " + "; ".join(issues))


class StageRunner(Generic[StageValue]):
    """Run invoke, parse, repair, and validation with a :class:`RetryPolicy`.

    The runner owns attempt counting and retry decisions.  Stage-specific
    prompts, parsing, repairs, and terminal validation errors stay in the
    caller's callbacks, keeping this abstraction intentionally small.
    """

    def __init__(self, *, retry_policy: RetryPolicy | None = None) -> None:
        self.retry_policy = retry_policy or RetryPolicy()
        self.last_failure: RetryContext | None = None
        self._failure_history: list[RetryContext] = []

    @property
    def failure_history(self) -> tuple[RetryContext, ...]:
        """Failures from the most recent :meth:`run`, in attempt order."""

        return tuple(self._failure_history)

    def run(
        self,
        *,
        invoke: Callable[[StageAttempt], StageValue],
        before_attempt: Callable[[StageAttempt], None] | None = None,
        parse: Callable[[StageValue, StageAttempt], StageValue] | None = None,
        repair: Callable[[StageValue, StageAttempt], StageValue] | None = None,
        validate: Callable[[StageValue, StageAttempt], Iterable[str] | None] | None = None,
        validation_error: Callable[[tuple[str, ...]], Exception] | None = None,
        on_failure: Callable[[RetryContext], None] | None = None,
        on_retry: Callable[[RetryContext], None] | None = None,
    ) -> StageValue:
        """Run one stage until it succeeds or its policy declines another try.

        Exceptions are retried only when ``retry_policy`` permits their phase.
        A final callback exception is re-raised with a bare ``raise`` so its
        original cause remains intact.  Validators return issue strings; their
        terminal exception is supplied by ``validation_error`` when needed.
        """

        self.last_failure = None
        self._failure_history.clear()
        for attempt_number in self.retry_policy.attempts():
            attempt = StageAttempt(attempt=attempt_number, max_attempts=self.retry_policy.max_attempts)
            if before_attempt is not None:
                before_attempt(attempt)
            try:
                value = invoke(attempt)
            except Exception as exc:
                context = self._record_failure(
                    attempt=attempt,
                    phase=RetryPhase.INVOKE,
                    error=exc,
                    on_failure=on_failure,
                )
                if self._schedule_retry(context, on_retry=on_retry):
                    continue
                raise

            if parse is not None:
                try:
                    value = parse(value, attempt)
                except Exception as exc:
                    context = self._record_failure(
                        attempt=attempt,
                        phase=RetryPhase.PARSE,
                        error=exc,
                        on_failure=on_failure,
                    )
                    if self._schedule_retry(context, on_retry=on_retry):
                        continue
                    raise

            if repair is not None:
                try:
                    value = repair(value, attempt)
                except Exception as exc:
                    context = self._record_failure(
                        attempt=attempt,
                        phase=RetryPhase.REPAIR,
                        error=exc,
                        on_failure=on_failure,
                    )
                    if self._schedule_retry(context, on_retry=on_retry):
                        continue
                    raise

            if validate is None:
                return value
            try:
                issues = self._normalize_issues(validate(value, attempt))
            except Exception as exc:
                context = self._record_failure(
                    attempt=attempt,
                    phase=RetryPhase.VALIDATE,
                    error=exc,
                    on_failure=on_failure,
                )
                if self._schedule_retry(context, on_retry=on_retry):
                    continue
                raise
            if not issues:
                return value

            error = (validation_error or StageValidationError)(issues)
            if not isinstance(error, Exception):
                raise TypeError("validation_error must return an Exception instance")
            context = self._record_failure(
                attempt=attempt,
                phase=RetryPhase.VALIDATION,
                error=error,
                issues=issues,
                on_failure=on_failure,
            )
            if self._schedule_retry(context, on_retry=on_retry):
                continue
            raise error

        raise RuntimeError("StageRunner exhausted without a terminal result")

    def _record_failure(
        self,
        *,
        attempt: StageAttempt,
        phase: RetryPhase,
        error: Exception,
        on_failure: Callable[[RetryContext], None] | None,
        issues: tuple[str, ...] = (),
    ) -> RetryContext:
        context = self.retry_policy.failure_context(
            attempt=attempt.attempt,
            phase=phase,
            error=error,
            issues=issues,
        )
        self.last_failure = context
        self._failure_history.append(context)
        if on_failure is not None:
            on_failure(context)
        return context

    def _schedule_retry(
        self,
        context: RetryContext,
        *,
        on_retry: Callable[[RetryContext], None] | None,
    ) -> bool:
        if not self.retry_policy.should_retry(context):
            return False
        if on_retry is not None:
            on_retry(context)
        return True

    @staticmethod
    def _normalize_issues(issues: Iterable[str] | None) -> tuple[str, ...]:
        if issues is None:
            return ()
        if isinstance(issues, str):
            return (issues,)
        return tuple(str(issue) for issue in issues)


__all__ = ["StageAttempt", "StageRunner", "StageValidationError"]
