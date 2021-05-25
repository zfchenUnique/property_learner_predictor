##### Scripts to run the experiments

    conda activate py37
    sh scripts/run_charge_dec_v2.sh ${GPU_ID}
    sh scripts/test_charge_dec_v2.sh ${GPU_ID}

##### Experiment folders

`log/exp_no_field_n_rollout_4_multi_msg_passing/`
 - batch size 16
 - no lr decay
 - multi-step message passing
 - multi-step future prediction during training

`exp_no_field_n_rollout_4_multi_msg_passing_B4_LRD8/`
 - batch size 8
 - lr decay with gamma 0.5 and epoch 8

`exp_no_field_n_rollout_4_multi_msg_passing_B4_LRD8_aug/`
 - batch size / lr decay same as the previous experiments
 - noise augmentation on the input x with std = 5e-3

`exp_no_field_n_rollout_4_multi_msg_passing_B4_LRD8_aug1e-3/`
 - noise augmentation on the input x with std = 1e-3

`exp_no_field_n_rollout_4_multi_msg_passing_B4_LRD8_aug2e-3/`
 - noise augmentation on the input x with std = 2e-3


