mass_pred_file='logs/exp_v16_mass_noise_001_prp/raw_prediction_mass_v16_test_prp.json' 
charge_pred_file='/home/zfchen/code/clevrer_dataset_generation_v2/models/NRI_CLEVRER/logs/exp_v16_charge_noise_001_prp/raw_prediction_charge_v16_test_prp.json'
obj_attr_dir='/home/zfchen/code/output/ns-vqa_output/v16_prp_test_refine/config'
output_dir='/home/zfchen/code/output/render_output_vislab3/v16_test/prediction_prp_mass_charge'
python tools/post_processing.py \
    --raw_result_mass_path  ${mass_pred_file} \
    --prp_config_dir ${obj_attr_dir} \
    --des_prp_dir ${output_dir} \
    --raw_result_charge_path ${charge_pred_file} \
