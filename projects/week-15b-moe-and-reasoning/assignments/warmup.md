# Warmup — read the router before you train it

1. Given `N` flattened tokens, `E` experts, and top-k `K`, derive the shape of
   router logits, all-expert probabilities, selected indices, and selected
   weights. For `K > 1`, explain why the selected softmax runs over `K`, not
   over the batch; for top-1, note that the selected all-expert probability is
   retained so task loss can train the router.
2. For `N=12`, `E=4`, `K=2`, and capacity factor `1.0`, calculate the per-expert
   capacity. What does it mean if 10 of the 24 selected assignments point to
   expert 0?
3. Write the Switch-style importance/load product in words. Why can a model
   have a finite task loss while a collapsed router gives one expert nearly
   all work?
4. Construct two groups of four arithmetic rewards, one with all equal rewards
   and one with one correct sample. Predict their relative advantages before
   running `group_relative_advantages`.
5. State the boundary between a verifier and a language model. What does a
   deterministic arithmetic verifier establish, and what does it not establish
   about open-ended reasoning?

**Start with a failing assertion:** check that every non-dropped selected
assignment has one expert index, that `K > 1` selected weights sum to one (and
top-1 weights equal the selected all-expert probabilities), and that a
constant-reward group receives zero advantages. Then run the tests.
