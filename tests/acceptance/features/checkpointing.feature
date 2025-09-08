Feature: Checkpoint retention policy enforcement
  As a user of ml_playground
  I want checkpoints to be managed according to retention policies
  So that disk space is optimized while preserving important model states

  Background:
    Given a fresh checkpoints directory

  Scenario: Keep policy enforcement for last checkpoints
    Given checkpoint retention policy of 2 last, 2 best
    When 3 checkpoints are saved sequentially
    Then 2 most recent checkpoints should exist
    And no stable last checkpoint pointer should exist

  Scenario: Keep policy enforcement for best checkpoints
    Given checkpoint retention policy of 2 last, 2 best
    When checkpoints are saved with the following metrics:
      | metric |
      | 1.0    |
      | 0.9    |
      | 1.1    |
    Then 2 best checkpoints by metric should exist
    And no stable best checkpoint pointer should exist

  Scenario: Filesystem discovery after restart
    Given checkpoint retention policy of 2 last, 1 best
    When 2 checkpoints are saved sequentially
    And the checkpoint manager is reinitialized
    Then existing checkpoints should be discovered

  Scenario: Skip last when best improves in same iteration
    Given checkpoint retention policy of 2 last, 2 best
    When an evaluation step produces improvement at iteration 0
    Then both best and last checkpoints should exist for iteration 0
    When an evaluation step produces improvement at iteration 10
    Then only best checkpoint should exist for iteration 10
