# HPD

This recipe keeps Hybrid Policy Distillation (HPD) as a recipe-local extension instead of patching core `verl`
PPO workers.

## Entry points

- `python3 -m recipe.HPD.main_hpd`
- `bash recipe/HPD/run_hpd.sh`

## Notes

- Current implementation targets `fsdp` / `fsdp2`.
- HPD currently expects `actor_rollout_ref.model.use_remove_padding=True`.
