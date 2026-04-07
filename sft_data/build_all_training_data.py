from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sft_data.intent_encoding.build_iea_sft_dataset import main as build_iea_main
from sft_data.intent_encoding.build_iea_chatml_sft_dataset import main as build_iea_chatml_main
from sft_data.optimization_strategy.build_osa_sft_dataset import main as build_osa_main
from sft_data.rl.build_rl_trace_dataset import main as build_rl_main
from sft_data.tool_call.build_warmup_dataset import main as build_warmup_main


def main() -> None:
    build_iea_main()
    build_iea_chatml_main()
    build_osa_main()
    build_warmup_main()
    build_rl_main()


if __name__ == "__main__":
    main()
