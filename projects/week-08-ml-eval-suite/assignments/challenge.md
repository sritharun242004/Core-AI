# Challenge — find the leak before the leaderboard

Create three deliberate failures: a target-copied feature, a preprocessing transform fitted before the split, and duplicate rows across train and validation. Use the suite plus a manual data audit to catch all three.

Extend the detector with a duplicate-row check and a time-aware split. Write a short incident report explaining why a perfect cross-validation score should have stopped the launch.
