# muzIQ Agent Instructions

## Training Target Guardrail

Long NSynth training runs must run on a G4 RTX Pro 6000 pod, not on local Mac
MPS. Treat `--epochs-per-stage >= 50`, full stage 1/2 training, or the
800/100/100 NSynth metadata split as long training.

Before starting long training:

1. Reserve or reuse an east5 spot G4 pod with
   `scripts/reserve-east5-g4-spot.sh`. Do not hand-write ad hoc G4 pod specs.
2. Verify the pod is `Running` with:
   `kubectl --context=east5 -n development get pod POD_NAME -o wide`
3. Verify the pod is actually an RTX Pro 6000 host with:
   `kubectl --context=east5 -n development exec POD_NAME -- nvidia-smi`
4. Start training through `scripts/train.sh --require-g4 ... --device cuda`.

Do not let a local Mac `--device mps` run become the primary run for long
training. A local run may only be a short fallback or smoke test, and must be
called out as local in progress updates.

Do not call `uv run python -m muziq_nn.training.train` directly for long
training. The `scripts/train.sh` wrapper rejects long runs that omit
`--require-g4`, rejects non-CUDA long runs by default, and verifies that
`--require-g4` is running on an RTX Pro 6000 GPU.

## east5 G4 Spot Pod Spec

Use `scripts/reserve-east5-g4-spot.sh` for G4 pod reservation. It uses
`kubectl --context=east5`, creates an opaque UUID pod name, and hard-codes the
east5 spot G4 selectors and tolerations below. Do not put project names such as
`muziq` in pod names.

The pod must include all of these selectors:

- `cloud.google.com/gke-accelerator=nvidia-rtx-pro-6000`
- `cloud.google.com/gke-spot=true`
- `node.kubernetes.io/instance-type=g4-standard-48`

The pod must tolerate all of these taints:

- `flyte.org/node-role=worker:NoSchedule`
- `union.ai/capacity-type=interruptible:NoSchedule`
- `union.ai/multithreading=disabled:NoSchedule`
- `nvidia.com/gpu=present:NoSchedule`
- `nvidia.com/gpu:NoSchedule`

If the pod is pending with `untolerated taint(s)`, inspect node taints with:

```bash
kubectl --context=east5 get nodes -o json |
  jq -r '.items[] | [.metadata.name,
    (.metadata.labels["node.kubernetes.io/instance-type"]//""),
    (.metadata.labels["cloud.google.com/gke-accelerator"]//""),
    ((.spec.taints//[])|map(.key+"="+(.value//"")+":"+.effect)|join(","))]
    | @tsv'
```

Fix the pod spec before waiting; do not wait on an under-tolerated pod. If an
existing pending pod is missing these selectors or tolerations, create a new
UUID-named pod with `scripts/reserve-east5-g4-spot.sh` instead of waiting on it.
