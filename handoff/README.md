# Handoff to a new machine

Prepared on 2026-11-10. Start with this file, then read `CONTEXT.md` and the
repository's current `AGENTS.md`. This directory supplements the Git checkout;
the maintained code, configs, papers, and existing documentation remain in their
normal locations. Originals were copied, not moved out of the working project.

## Contents

- `CONTEXT.md`: current research question, implementation map, evidence, and open decisions.
- `conversation.jsonl.gz`: the full available local Codex session transcript,
  compressed without summarizing or editing its records. It includes earlier
  compactions as recorded by Codex, rather than reconstructing information that
  was never stored. The snapshot ends during packaging; later messages are absent.
- `forge-source.tar.gz`: the exact tracked sibling Forge source at the revision
  in `manifest.json`, including its packaging metadata and README.
- `experiment-artifacts.tar.gz`: Forge configs, run metadata, results, scalar
  logs, available canonicalization metadata, current plots/reports, small shared
  benchmark checkpoints, and available Berman MNIST input gzip files. Archive
  members retain their `outputs/...` paths.
- `runs.json`: an index of the archived Forge runs, including recorded status and
  accepted endpoints where readable. A manually interrupted run can have failed
  Forge metadata while its last `result.json` still says running.
- `manifest.json`: source revisions, installed package versions, transcript
  destination, bundled artifact paths, and the large checkpoints omitted from Git.
- `AGENTS.snapshot.md`: the repository instructions at packaging time. The current
  root `AGENTS.md` remains authoritative.
- `SHA256SUMS`: integrity hashes for the other files in this directory.

The transcript contains this private working conversation and tool outputs; treat
the bundle as project data. Codex authentication files and other sessions are not
included. No separate user memory files existed in `~/.codex/memories/` at packaging.

## Restore the project

From the freshly cloned pruning repository, with no existing sibling `forge/`
checkout that would be overwritten:

```bash
sha256sum -c handoff/SHA256SUMS
tar -xzf handoff/forge-source.tar.gz -C ..
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -e ../forge
tar -xzf handoff/experiment-artifacts.tar.gz -C .
```

The Forge archive is a source snapshot, without `.git`. Alternatively clone
`https://github.com/simonelusetti/forge.git` and check out the revision in the
manifest. Runtime device and CPU thread count must be selected for the new host.
The installed versions recorded in the manifest describe the old host; they are
not a portable environment image. CUDA compatibility and a Gurobi license, if
using the original MIQP method, still depend on the new machine.

Hugging Face OPT/C4 caches are not bundled. The new machine needs network access
to download them, or a separate cache transfer. Do not copy the old run's
`HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` settings until those caches exist.

## Restore the conversation

Install Codex and sign in on the new machine, then:

```bash
mkdir -p ~/.codex/sessions/2026/10/22
gzip -dc handoff/conversation.jsonl.gz > ~/.codex/sessions/2026/10/22/rollout-2026-10-22T00-42-16-01a160a1-26a9-7973-ab29-5eb3aa93370c.jsonl
codex resume 01a160a1-26a9-7973-ab29-5eb3aa93370c --cd "$PWD"
```

Do not overwrite an existing copy of this session containing newer messages.
`codex resume` and its directory override are supported CLI features; copying
session files across machines is a manual migration and VS Code discovery has
not been verified. If resuming fails, give a new agent `CONTEXT.md` and access to
the decompressed transcript. Read it selectively rather than filling context
with the entire history. Verify continuity before removing the old machine.

## Large artifacts

About 11.8 GB of per-run model checkpoints are excluded from this Git bundle.
Their paths and sizes are listed in `manifest.json`. They are needed for
checkpoint-dependent analyses or continuation, but not for inspecting the saved
curves or starting a new experiment. To transfer them from the old host, run from
the new repository, replacing the host and source path as appropriate:

```bash
rsync -av --include='*/' --include='*.pt' --include='*.pth' --include='*.safetensors' --exclude='*' OLD_HOST:/mnt/data/workspace_simonelusetti/pruning/outputs/xps/ outputs/xps/
```

The last fixed-penalty run has **no saved model checkpoint**: it was interrupted
before the final checkpoint write. Its recorded results are included. This
bundle cannot reconstruct that exact model state from the result metadata alone.

## Plot caveat

The bundled OPT image includes a manually plotted accepted snapshot of the
interrupted fixed-penalty run at step 82 (5.7118% eligible removal). The run later
reached step 100 (5.7305%). Default plotting excludes failed runs, so regenerating
the image normally drops that point. Preserve the bundled image or explicitly
load that run's accepted result if reproducing this comparison. Do not relabel
its metadata as completed merely to make the plot include it.

Existing supervisor reports are historical artifacts, not current implementation
specifications. Use the code, resolved configs, and saved results to settle
disagreements.
