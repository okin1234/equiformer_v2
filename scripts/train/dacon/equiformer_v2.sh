python -u -m torch.distributed.launch --nproc_per_node=4 main_oc20.py \
    --distributed \
    --num-gpus 4 \
    --mode train \
    --config-yml 'oc20/configs/dacon/equiformer_v2.yml' \
    --run-dir 'models/dacon/equiformer_v2/test_v1' \
    --print-every 200 \
    --amp
