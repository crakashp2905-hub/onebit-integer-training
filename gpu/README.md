# GPU runs on Kaggle

**The directory is `gpu/`, not `kaggle/`, deliberately.** A top-level `kaggle/`
directory shadows the installed `kaggle` package on every `import kaggle` run
from the repo root. The original layout hit exactly that and it is not worth
rediscovering.

```
python gpu/push.py --configs R6_attn8,R7_head8 --steps 10000
python gpu/watch.py                 # poll until COMPLETE / ERROR / CANCEL
python gpu/pull.py                  # fetch output into results/gpu3/
```

## Everything created here is PRIVATE

`push.py` sets `isPrivate: true` on both the dataset and the kernel, and
refuses to push if that flag is not set. It also runs a secret scan over the
payload before uploading anything, and aborts on a hit.

## Things that were learned the hard way

- **The mount path is not what the docs imply.** It is
  `/kaggle/input/datasets/<user>/<dataset>/`, not `/kaggle/input/<dataset>/`.
  `entry.py` prints the whole input tree and then *searches* for the repo root
  rather than assuming a path.
- **The mount is read-only.** Tokenizer and batch caches must go somewhere
  writable, hence `ONEBIT_CACHE`; results likewise, hence `ONEBIT_RESULTS`.
- **A watcher must not treat unknown output as terminal.** An earlier shell
  version used a `case` statement whose default branch meant "finished", so a
  single DNS blip reported the run as complete. Here only the three real
  terminal states end the loop; every exception is transient and retried.
- **Batch 16 x ctx 128 = 2048 tokens/step does not fill a GPU.** Measured only
  1.74x over 4 CPU cores on a P100 -- kernel-launch overhead across many tiny
  fake-quant elementwise ops, not arithmetic. Raising the batch would fix it,
  but it also changes the replayed batch sequence, so it cannot be done to a
  run that must stay comparable with existing results. It is a change for a
  fresh budget, not a retrofit.
