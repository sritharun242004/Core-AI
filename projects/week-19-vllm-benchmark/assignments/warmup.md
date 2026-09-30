# Warmup — count the intervals (30 minutes)

1. Before running the notebook, calculate TTFT, TPOT, and latency for submission 0,
   output observations (0.2,0.3,0.4), completion 0.5 seconds. Add a one-token request
   submitted 0.1, token at 0.6, completion 0.8. Use a [0,1] wall window for throughput.
2. Explain why one-token TPOT is absent rather than zero and why throughput includes
   first output tokens. Add an empty successful response and state which denominators
   and sample counts change.
3. Quantize `[-1,0,0.5,1]` with INT4, group size 4. List qmax, scale, codes,
   reconstruction, and max error. Repeat with INT8 and then two groups of two.
4. Compute ideal storage for 128 INT4 weights plus two float32 scales. Explain why
   the reference array's physical storage differs.

**Deliver:** a one-page table with units, handwritten arithmetic, and the notebook
output labelled `simulation_not_hardware_measurement`. No GPU or downloads.

**Acceptance:** correct n−1 TPOT denominator, absent undefined metrics, wall-window
throughput, error at most scale/2 (within float roundoff), and metadata accounted for.
Consult `SOLUTION_NOTES.md` only after writing your answers.
