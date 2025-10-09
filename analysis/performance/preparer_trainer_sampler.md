# Performance Review: Preparer, Trainer, and Sampler Workflows

This report highlights hotspots where the current implementations rely on Python-level loops or data wrangling that could be replaced by faster primitives from vectorized libraries (NumPy/PyTorch) or specialized tokenization utilities.

## Preparer Workflow

- **Vocabulary construction in `prepare_with_tokenizer`** – The char/word branches rebuild vocabularies with `sorted(set(...))` and dictionary comprehensions over Python iterables.【F:ml_playground/data_pipeline/transforms/tokenization.py†L39-L55】  On large corpora this becomes a Python-bound (O(n log n)) routine.
  - *Current complexity*: `sorted(set(...))` requires de-duplicating (`O(n)`) and sorting (`O(n log n)`), so the dominant cost is `O(n log n)` per corpus scan, repeated whenever the function is called.
  - *Proposed complexity*: Using `numpy.unique`, `pandas.factorize`, or tokenizer libraries keeps the algorithmic work at `O(n)` expected time by avoiding the explicit sort and delegating to hashed/streaming implementations in optimized native code, yielding both better asymptotics and constant-factor gains.
  - *Net change*: From `O(n log n)` Python-level processing to `O(n)` native-backed processing per corpus.

## Trainer Workflow

- **Random batch sampling path** – When `L <= block_size`, `_take_seq` builds each training example inside a Python list comprehension, invoking NumPy array creation `batch_size` times and performing wrap-around in Python.【F:ml_playground/data_pipeline/sampling/batches.py†L33-L51】  Even in the `L > block_size` branch we repeatedly slice in a comprehension.【F:ml_playground/data_pipeline/sampling/batches.py†L52-L62】  Replacing those comprehensions with vectorized indexing (`np.take`, `sliding_window_view`) lets NumPy handle the copying in compiled code.
  - *Current complexity*: Each batch materialization performs `batch_size` Python iterations that each copy `block_size` tokens, i.e., `O(batch_size × block_size)` work with high interpreter overhead.
  - *Proposed complexity*: Building an index matrix once and using `np.take` or `sliding_window_view` executes the same `O(batch_size × block_size)` data movement inside NumPy's native loops, reducing Python involvement to `O(1)` setup per batch.
  - *Net change*: Asymptotic work stays `O(batch_size × block_size)` but the Python-layer cost drops from `O(batch_size)` control overhead to `O(1)` by vectorizing the operation.
- **Sequential sampler implementation** – `_get_sequential_batch` appends each sequence in a Python `for` loop, juggling multiple branches to handle wrap-around and concatenating fragments manually.【F:ml_playground/data_pipeline/sampling/batches.py†L125-L173】  Constructing an index grid with `np.arange` and reshaping or using `sliding_window_view` can eliminate the per-example Python loop, producing the full `bsz × T` batch in one vectorized call.
  - *Current complexity*: `O(batch_size × block_size)` Python iterations with additional branching for wrap-around, each iteration copying slices.
  - *Proposed complexity*: Generating an index grid or sliding window delegates the same `O(batch_size × block_size)` data movement to compiled NumPy kernels with `O(1)` Python coordination.
  - *Net change*: No asymptotic change, but significant reduction in interpreter overhead and better cache-friendly access.
- **Gradient accumulation loop** – `_train_step` performs gradient accumulation with a pure-Python loop over `grad_accum_steps`.【F:ml_playground/training/loop/runner.py†L285-L305】  When accumulation steps are high, this loop becomes Python-bound.  Leveraging PyTorch utilities such as `torch.vmap`/`torch.compile` or fused gradient-accumulation kernels (e.g., through `accelerate` or custom CUDA ops) would reduce Python overhead per micro-step.
  - *Current complexity*: Each training step triggers `O(grad_accum_steps)` Python iterations that each launch a backward pass, layering interpreter overhead on top of the required `O(grad_accum_steps)` tensor work.
  - *Proposed complexity*: Fusing the accumulation via `torch.vmap`, `torch.compile`, or library-provided fused kernels keeps the tensor work at `O(grad_accum_steps)` but collapses Python coordination to `O(1)` per step.
  - *Net change*: Maintains the necessary tensor complexity while shaving Python-loop overhead from linear in `grad_accum_steps` to constant.

## Sampler Workflow

- **Prompt tensor construction** – Converting the starting prompt IDs with `torch.tensor(...)[None, ...]` materializes a new tensor from a Python list every run.【F:ml_playground/sampling/runner.py†L167-L176】  Using `torch.as_tensor` (which can share memory) or precomputing prompts as device tensors avoids repeated Python-to-C copies when sampling multiple prompts.
  - *Current complexity*: `torch.tensor(list_ids)` is `O(prompt_length)` every invocation with Python copying.
  - *Proposed complexity*: Reusing cached tensors or calling `torch.as_tensor` amortizes the cost to `O(1)` for repeated prompts (or keeps it at `O(prompt_length)` but in native code when new data is needed).
  - *Net change*: Potentially reduces repeated runs from `O(prompt_length)` Python work to `O(1)` reuse, or at least shifts the cost into native kernels.
- **Decoding generated tokens** – Each sample calls `y[0].tolist()` before decoding, incurring a Python loop over every generated token.【F:ml_playground/sampling/runner.py†L180-L189】  If the tokenizer accepts NumPy arrays or PyTorch tensors, routing through `y[0].cpu().numpy()` or a vectorized decoder (from libraries like `tokenizers`) would shift the heavy conversion work into optimized code and cut Python overhead for long generations.
  - *Current complexity*: `tolist()` walks all `O(generated_tokens)` elements in Python space per sample.
  - *Proposed complexity*: Vectorized decoding keeps the `O(generated_tokens)` traversal inside C/Rust code (e.g., via tokenizer bindings) or reuses streaming decoders, leaving only `O(1)` Python orchestration per sample.
  - *Net change*: Same asymptotic data size but eliminates the Python-level `O(generated_tokens)` work.

---
Prioritizing these areas should yield noticeable wins because they sit directly on the critical paths for dataset preparation, batch delivery during training, and text generation.
