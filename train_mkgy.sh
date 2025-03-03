data=MKG-Y
num_epoch=5000
hidden_dim=1024
lr=5e-4
dim=128
max_vis_token=8
max_txt_token=25
num_layer_dec=2
cuda_device=4

CUDA_VISIBLE_DEVICES=${cuda_device} python train_rs2c.py --data ${data} --num_epoch ${num_epoch}\
                                                        --hidden_dim ${hidden_dim} --lr ${lr} --dim ${dim}\
                                                        --max_vis_token ${max_vis_token} --max_txt_token ${max_txt_token}\
                                                        --num_layer_dec ${num_layer_dec} 
