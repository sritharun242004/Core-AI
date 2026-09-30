# Compute — red for the real scaling extension

All shipped tests/notebooks use tiny CPU fixtures at $0 cloud cost. The optional three-size Transformer/TinyStories sweep is a separate experiment requiring an approved budget. Example planning envelope: RunPod A100 40GB or a comparable provider instance, 3 sizes × 3 seeds × 0.5–2 hours = 4.5–18 GPU-hours. Multiply a current quote by those hours and add storage/egress; $20–60 is only a provisional ceiling for a carefully reduced sweep, not a guarantee. If the quote exceeds the ceiling, reduce the workload before launch.

Prepare licensed data and tokenizer locally, pin the environment and test a single batch before scheduling. Track actual tokens and elapsed time, stop at the predeclared budget and report incomplete experiments honestly. Save artifacts, terminate every instance/pod, delete unused volumes and inspect billing. No cloud experiment or dataset download is performed by this project.
