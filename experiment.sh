#!/bin/bash

set -euo pipefail

usage() {
	cat <<'EOF'
Usage: bash experiment.sh [--model deepseek|qwen] [--scenario S2|S3] [--method B|B1|B2|B3|Ours|ablation]

Examples:
  bash experiment.sh
	bash experiment.sh --model qwen --scenario S2 --method B
	bash experiment.sh --model qwen --scenario S2 --method B2
  bash experiment.sh --scenario S3 --method Ours
  bash experiment.sh --model deepseek --method ablation

Notes:
	- --method B runs all baseline methods: B1, B2, B3.
	- --method B1, B2, or B3 runs only that baseline.
  - DRY_RUN=1 prints matching commands without executing them.
EOF
}

model_filter=""
scenario_filter=""
method_filter=""

while [[ $# -gt 0 ]]; do
	case "$1" in
		--model)
			[[ $# -ge 2 ]] || { echo "Missing value for --model" >&2; exit 1; }
			model_filter="$2"
			shift 2
			;;
		--scenario)
			[[ $# -ge 2 ]] || { echo "Missing value for --scenario" >&2; exit 1; }
			scenario_filter="$2"
			shift 2
			;;
		--method)
			[[ $# -ge 2 ]] || { echo "Missing value for --method" >&2; exit 1; }
			method_filter="$2"
			shift 2
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			echo "Unknown argument: $1" >&2
			usage >&2
			exit 1
			;;
	esac
done

case "$model_filter" in
	""|deepseek|qwen)
		;;
	*)
		echo "Invalid --model: $model_filter" >&2
		usage >&2
		exit 1
		;;
esac

case "$scenario_filter" in
	""|S2|S3)
		;;
	*)
		echo "Invalid --scenario: $scenario_filter" >&2
		usage >&2
		exit 1
		;;
esac

case "$method_filter" in
	"")
		;;
	B|B1|B2|B3|Ours|ablation)
		;;
	*)
		echo "Invalid --method: $method_filter" >&2
		usage >&2
		exit 1
		;;
esac

experiments=(
	"E1|S2|Ours|default"
	"E3|S2|Ours_wo_ClosedLoop|default"
	"E4|S2|Ours_wo_RAG|default"
	"E1|S2|Ours|deepseek"
	"E1|S2|B1|default"
	"E1|S2|B2|default"
	"E1|S2|B3|default"
	"E1|S2|B1|qwen"
	"E1|S2|B2|qwen"
	"E1|S2|B3|qwen"
	"E1|S3|Ours|default"
	"E1|S3|B3|default"
	"E1|S3|B3|qwen"
)

selected_count=0

for spec in "${experiments[@]}"; do
	IFS='|' read -r experiment scenario method model <<< "$spec"

	case "$method" in
		B1|B2|B3)
			method_group="B"
			;;
		Ours)
			method_group="Ours"
			;;
		Ours_wo_ClosedLoop|Ours_wo_RAG)
			method_group="ablation"
			;;
		*)
			echo "Unsupported method in experiment list: $method" >&2
			exit 1
			;;
	esac

	if [[ -n "$model_filter" && "$model" != "$model_filter" ]]; then
		continue
	fi

	if [[ -n "$scenario_filter" && "$scenario" != "$scenario_filter" ]]; then
		continue
	fi

	if [[ -n "$method_filter" ]]; then
		if [[ "$method_filter" == "B" ]]; then
			[[ "$method_group" == "B" ]] || continue
		else
			[[ "$method" == "$method_filter" || "$method_group" == "$method_filter" ]] || continue
		fi
	fi

	cmd=(python experiments/scripts/launch_experiments.py --experiment "$experiment" --scenario "$scenario" --method "$method")
	if [[ "$model" == "deepseek" ]]; then
		cmd+=(--deepseek)
	elif [[ "$model" == "qwen" ]]; then
		cmd+=(--qwen)
	fi

	((selected_count += 1))
	echo "[$selected_count] ${cmd[*]}"

	if [[ "${DRY_RUN:-0}" == "1" ]]; then
		continue
	fi

	"${cmd[@]}"
done

if [[ $selected_count -eq 0 ]]; then
	echo "No experiments matched the provided filters." >&2
	exit 1
fi