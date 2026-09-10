# M4 Final Integration Audit

## Status

M4 core orchestration is implemented and tested. Final integration must be validated against the real M1/M2/M3/M5 specialist interfaces before declaring production integration complete.

## Contract

M4 owns the stable `ToolResult` contract and specialist adapters. Real specialist implementations remain owned by their module directories.

## Final validation targets

- M1 EarthDial/VQA → `m1_vqa`
- M2 change detection → `m2_change_detection`
- M2 grounding → `m2_grounding`
- M3 optical/SAR → `m3_optical_sar`
- M5 GIS → `m5_gis`

## Acceptance criteria

1. Natural-language query is parsed and classified.
2. Planner creates the correct specialist sequence.
3. Executor resolves runtime inputs and previous results.
4. Registry invokes real adapters.
5. Every adapter normalizes external output to `ToolResult`.
6. Synthesizer produces one `AgentResponse`.
7. Missing specialists/inputs fail safely rather than crashing the controller.
8. Full regression suite passes.
9. At least one real specialist-backed integration path is exercised where the environment permits.

The audit deliberately does not claim that heavyweight ML inference has been executed in CI; model/GPU/data availability is an environment concern.