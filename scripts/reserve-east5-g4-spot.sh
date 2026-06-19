#!/usr/bin/env bash
set -euo pipefail

context="east5"
namespace="development"
image="nvcr.io/nvidia/pytorch:25.04-py3"
hours="72"
dry_run="false"

usage() {
  cat <<'EOF'
Usage: scripts/reserve-east5-g4-spot.sh [--hours HOURS] [--image IMAGE] [--dry-run]

Creates a UUID-named east5 spot G4 RTX Pro 6000 pod with the required
selectors and tolerations. Pod names are intentionally opaque UUIDs. The
default lifetime is 72 hours.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hours)
      hours="$2"
      shift 2
      ;;
    --image)
      image="$2"
      shift 2
      ;;
    --dry-run)
      dry_run="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$hours" =~ ^[0-9]+$ ]] || [[ "$hours" -lt 1 ]]; then
  echo "--hours must be a positive integer" >&2
  exit 2
fi

if command -v uuidgen >/dev/null 2>&1; then
  pod_name="$(uuidgen | tr '[:upper:]' '[:lower:]')"
else
  pod_name="$(python3 -c 'import uuid; print(uuid.uuid4())')"
fi

seconds=$((hours * 3600))
overrides="$(cat <<JSON
{
  "spec": {
    "restartPolicy": "Never",
    "nodeSelector": {
      "cloud.google.com/gke-accelerator": "nvidia-rtx-pro-6000",
      "cloud.google.com/gke-spot": "true",
      "node.kubernetes.io/instance-type": "g4-standard-48"
    },
    "affinity": {
      "nodeAffinity": {
        "requiredDuringSchedulingIgnoredDuringExecution": {
          "nodeSelectorTerms": [
            {
              "matchExpressions": [
                {
                  "key": "flyte.org/node-role",
                  "operator": "In",
                  "values": ["worker"]
                }
              ]
            }
          ]
        }
      }
    },
    "tolerations": [
      {
        "key": "flyte.org/node-role",
        "operator": "Equal",
        "value": "worker",
        "effect": "NoSchedule"
      },
      {
        "key": "union.ai/capacity-type",
        "operator": "Equal",
        "value": "interruptible",
        "effect": "NoSchedule"
      },
      {
        "key": "union.ai/multithreading",
        "operator": "Equal",
        "value": "disabled",
        "effect": "NoSchedule"
      },
      {
        "key": "nvidia.com/gpu",
        "operator": "Equal",
        "value": "present",
        "effect": "NoSchedule"
      },
      {
        "key": "nvidia.com/gpu",
        "operator": "Exists",
        "effect": "NoSchedule"
      }
    ],
    "containers": [
      {
        "name": "gpu",
        "image": "$image",
        "command": ["sleep", "$seconds"],
        "resources": {
          "requests": {
            "cpu": "46",
            "memory": "150G",
            "ephemeral-storage": "35G",
            "nvidia.com/gpu": "1"
          },
          "limits": {
            "cpu": "48",
            "memory": "160G",
            "ephemeral-storage": "40G",
            "nvidia.com/gpu": "1"
          }
        },
        "env": [
          {
            "name": "NVIDIA_VISIBLE_DEVICES",
            "value": "all"
          }
        ],
        "securityContext": {
          "privileged": true
        }
      }
    ]
  }
}
JSON
)"

kubectl_args=(
  --context="$context"
  --namespace="$namespace"
  run "$pod_name"
  --image="$image"
  --restart=Never
  --overrides="$overrides"
)

if [[ "$dry_run" == "true" ]]; then
  kubectl_args+=(--dry-run=client -o yaml)
fi

kubectl "${kubectl_args[@]}"

if [[ "$dry_run" == "false" ]]; then
  echo
  echo "Created pod: $pod_name"
  echo "Inspect:"
  echo "  kubectl --context=$context -n $namespace get pod $pod_name -o wide"
  echo "  kubectl --context=$context -n $namespace describe pod $pod_name"
  echo "Verify GPU after Running:"
  echo "  kubectl --context=$context -n $namespace exec $pod_name -- nvidia-smi"
fi
