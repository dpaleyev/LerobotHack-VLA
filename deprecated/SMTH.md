## Всякая всячина для запуска

leader port = /dev/ttyACM0

$ ./run_official_smolvla_train_cached.sh --policy.path=/app/outputs/models/lerobot_smolvla_base_sanitized_lr1e4 --dataset.repo_id=so101_pnp --dataset.root=/app/demo_data_merged_draft_hf --batch_size=32 --steps=20000 --output_dir=/app/outputs/train/so101_smolvla_official_main_bs32_lr1e4_noamp --job_name=so101_smolvla_official_main_bs32_lr1e4_noamp --policy.device=cuda --policy.use_amp=false --wandb.enable=false --log_freq=50 --save_freq=1000 --num_workers=4