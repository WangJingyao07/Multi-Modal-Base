for alpha1 in {0.4,0.5,0.6,0.7,0.8,0.9}
do
    for alpha2 in {0.9,0.8,0.7,0.6,0.5}
    do
        for alpha3 in {0.1,0.2,0.3,0.4}
        do 
        task="IEMOCAP" 
        task_type="classification"
        model="mmrg"
        name=${task}"_"${model}"_"${alpha1}"_"${alpha2}"_"${alpha3}"_if"
        echo alpha1: $alpha1 alpha2: $alpha2 alpha3: $alpha3
        CUDA_VISIBLE_DEVICES=3 python train_IEMO_MMRG.py --batch_size 16 --gradient_accumulation_steps 40 --savedir ./saved/$task --name $name  \
        --alpha1 $alpha1 --alpha2 $alpha2 --task IEMOCAP --task_type classification --model mml_avt_mlu \
        --patience 5 --lr 5e-05 --seed 1 --noise 0.0 --alpha3 $alpha3
        done
    done
done