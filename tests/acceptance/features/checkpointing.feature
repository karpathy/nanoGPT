Feature: Checkpointing policy and rotation (strict, rotated-only)
  As a user of ml_playground
  I want checkpoints to rotate and enforce keep policies without stable pointers
  So that only rotated checkpoints are produced and used

  Scenario: Keep policy enforcement for last checkpoints
    Given a fresh checkpoints directory
    And a checkpoint manager with keep_last 2 and keep_best 2
    When I save 3 last checkpoints sequentially
    Then only the 2 most recent last checkpoints exist
    And no stable last pointer exists

  Scenario: Keep policy enforcement for best checkpoints
    Given a fresh checkpoints directory
    And a checkpoint manager with keep_last 2 and keep_best 2
    When I save 3 best checkpoints with metrics 1.0, 0.9, 1.1
    Then only the 2 best checkpoints by metric exist
    And no stable best pointer exists

  Scenario: Filesystem discovery after restart
    Given a fresh checkpoints directory
    And a checkpoint manager with keep_last 2 and keep_best 1
    When I save 2 last checkpoints sequentially
    And I reinitialize the checkpoint manager
    Then the manager discovers the existing last checkpoints

  Scenario: Skip last when best improves in same iteration
    Given a fresh checkpoints directory
    And a checkpoint manager with keep_last 2 and keep_best 2
    When I simulate an eval step with an improvement at iter 0
    Then both best and last checkpoints exist for iter 0
    When I simulate an eval step with an improvement at iter 10
    Then only a best checkpoint exists for iter 10 (no last in same step)
