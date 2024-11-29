set -e -u

# set env `FLUX_DEV=xxx/flux1.safetensors` and `AE=xxx/ae.safetensors` if you want to use local flux
echo "flux path: $FLUX_DEV"
echo "ae   path: $AE"

python -u src/edit.py \
    --source_prompt "A young boy is playing with a toy airplane on the grassy front lawn of a suburban house, with a blue sky and fluffy clouds above." \
    --target_prompt "A young boy is playing with a toy airplane on the grassy front lawn of a suburban house, with a small yellow dog playing beside him, and a blue sky with fluffy clouds above." \
    --guidance 2 \
    --source_img_dir 'imgs/boy.jpg' \
    --num_steps 15 \
    --offload \
    --inject 1 \
    --name 'flux-dev'  \
    --output_dir 'edit-result/adddog1' 


