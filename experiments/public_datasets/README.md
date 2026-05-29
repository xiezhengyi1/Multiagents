# Public Dataset Driven Profiles

## Scope

This directory stores public datasets, derived traffic profiles, and scenario variants aligned to the three experiment scenes.

## Dataset Table

| Dataset family | Public source | Role in this project |
|---|---|---|
| Instant messaging video calls | `https://zenodo.org/records/8006901` | Proxy for bidirectional low-latency visual interaction flows |
| HTTPS traffic classification | `https://zenodo.org/records/4911551` | Proxy for video-player traffic and web-browsing traffic |
| VR AR CG telemetry | `https://github.com/dcomp-leris/VR-AR-CG-network-telemetry` | Proxy for cloud-gaming style control and rendered-media flows |
| FANET UAV dataset | `https://zenodo.org/records/19373220` | Proxy for UAV related control traffic |
| LoRaWAN traffic analysis | `https://zenodo.org/records/8090619` | Proxy for sparse IoT telemetry flows |

## Output Table

| Artifact | Meaning |
|---|---|
| `raw/` | Downloaded public dataset files used in local extraction |
| `derived/profile_catalog.json` | Extracted statistics, source metadata, and flow-to-profile mapping |
| `../scenarios_public/` | Scenario variants produced from the extracted profiles |

## Rebuild

| Command | Function |
|---|---|
| `python experiments/scripts/build_public_dataset_profiles.py` | Recompute profile statistics and regenerate public-dataset-driven scenario YAML files |

## Modeling Notes

| Topic | Explanation |
|---|---|
| Semantic mismatch | Several project flow names such as `Remote_Drive` and `Factory_Robot` do not have one-to-one public packet datasets in the downloaded sources. The generated profiles therefore use the nearest public traffic family and record the approximation in `profile_catalog.json`. |
| Unsupported fields | Some public datasets expose packet and bandwidth statistics but do not expose explicit end-to-end latency or jitter. In those cases the generator updates only the supported traffic fields and preserves the original SLA entries. |
| Packetization assumption | The instant-messaging video-call dataset provides aggregate throughput, queueing delay, and loss statistics. Its packet-size fields in the derived profile therefore remain an explicit packetization assumption documented in the profile catalog. |
