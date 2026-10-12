"""Development-set validation follows the optimizer-step budget."""

from training.schedule import (
    during_training_steps,
    resolve_training_schedule,
    validation_interval,
    validation_steps,
)


def test_one_thousand_steps_validate_every_hundred() -> None:
    assert validation_interval(1000, 10) == 100
    assert validation_steps(1000, 10) == [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def test_interval_scales_with_the_step_budget() -> None:
    assert validation_interval(500, 10) == 50
    assert validation_steps(500, 10)[-1] == 500
    assert validation_interval(10, 10) == 1
    # A budget that does not divide evenly still scores the final step.
    assert validation_steps(999, 10)[0] == 99
    assert validation_steps(999, 10)[-1] == 999


def test_during_training_checks_are_thirty_and_seventy_percent() -> None:
    assert during_training_steps(1000) == [300, 700]
    assert during_training_steps(10) == [3, 7]
    # 30% of 7 is step 2 and 70% is step 5. Step 7 is the evaluation after training.
    assert during_training_steps(7) == [2, 5]
    assert during_training_steps(2) == [1]
    assert during_training_steps(1) == []


def test_schedule_uses_epochs_when_steps_are_unset() -> None:
    schedule = resolve_training_schedule(
        train_count=80,
        batch_size=8,
        epochs=1,
        steps=None,
        val_checks=10,
    )
    assert schedule["optimizer_steps"] == 10
    assert schedule["validation_steps"] == [3, 7]
    assert schedule["validation_after_training"] is True
    assert schedule["val_checks"] == 3
    assert schedule["frozen_used_for_selection"] is False


def test_explicit_steps_override_epochs() -> None:
    schedule = resolve_training_schedule(
        train_count=80,
        batch_size=8,
        epochs=10,
        steps=1000,
        val_checks=10,
    )
    assert schedule["optimizer_steps"] == 1000
    assert schedule["validation_steps"] == [300, 700]
    assert schedule["validation_after_training"] is True
    assert schedule["frozen_used_for_selection"] is False
