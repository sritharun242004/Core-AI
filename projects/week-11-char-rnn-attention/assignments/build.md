# Build — inspect the seq2seq contract (2–3 hrs)

1. Run the offline fixture through `CharLSTMAttention`. Print logits and
   attention shapes, then plot or print one attention row for a prompt.
2. Compare teacher-forced loss with free-running generation at a fixed prompt.
   Keep the seed, vocabulary, sequence length, and decoding temperature fixed.
3. Add an explicit BOS/EOS token pair to a local branch and update the failing
   shape tests before changing the model. Explain whether the extra symbols
   alter the target shift or only the vocabulary.
4. Train on the deterministic fixture for a few epochs and report token loss,
   gradient norm, and two generated samples. Do not call the fixture
   Shakespeare and do not download data in a test.

**Deliverable:** a short experiment table with configuration, final loss, and
samples, followed by a paragraph distinguishing teacher forcing from
free-running quality.
