for Temp in {0.1,0.5,1,5,7,10}
do
    task="IEMOCAP" 
    task_type="classification"
    name=${task}"_"${Temp}"_{T*T}"
    echo temp: $Temp
    CUDA_VISIBLE_DEVICES=1 python ../train/train_IEMO_D.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/IEMOCAP_D --name $name --task IEMOCAP --task_type classification --model mml_avt  --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --max_epochs 100 --Temp $Temp
done