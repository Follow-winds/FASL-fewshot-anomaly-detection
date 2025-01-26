# config
MODEL_NAME=$1                                         # model name for train/test results
DEVICE=$2                                             # device: cpu , cuda number
USE_NAM=$3                                            # use or not use "Neighbor Aggregation Module"
WIN_MODE=$4                                           # [test] option: nan , mul , onlywin
WIN_SIZE=$5                                           # [test] default: 13 patchs
WIN_STRIDE=$6                                         # [test] default: 2 patchs
WIN_WEIGHT=$7                                         # [test] default: 0.5
SEED=$8                                               # random seed
KSHOTS=$9                                             # k-shots

# ============================================================================================================

model_name=${MODEL_NAME}                              # model name
useadapter=0                                          # use adapter?
useSENet=0                                            #
useabs=1                                              # use diff abs?
useFirstBatchNorm=1                                   #
useConvRes=${USE_NAM}                                 # use NAM?
usecrop=1                                             # visa crop?
useZeroCls=0                                          # use zero cls?
ResFusionMode=nan                                     # zero res fusion mode
testMode=fewshot                                      # test mode: zeroshot fewshot
useimgidx=0                                           # use imgidx mem bank

winmode=${WIN_MODE}                                   # nan mul onlywin
winuseConvRes=1
winsize=${WIN_SIZE}
winZipSize=${WIN_SIZE}
winstride=${WIN_STRIDE}
usewinmem=1
usewinimgencoder=1
winwight=${WIN_WEIGHT}

device=${DEVICE}                                      # cuda
testnumber=15                                         #
epochs=15
onvisa_epochs=11
onmvtec_epochs=5
testvisa_epochs=11
testmvtec_epochs=5
seed=${SEED}
count_res=0                                           #
alignMode=mulmax                                      # diff2 mulmax
useNorm=1

testsaveimg=0                                         # test save img

echo "model name: ${model_name}"
echo "use adapter: ${useadapter}"
echo "use diff abs: ${useabs}"
echo "useFirstBatchNorm: ${useFirstBatchNorm}"
echo "use NAM: ${useConvRes}"
echo "useSENet: ${useSENet}"
echo "use visa crop: ${usecrop}"
echo "use zero cls: ${useZeroCls}"
echo "zeroResFusionMode: ${ResFusionMode}"
echo "mode: ${testMode}"
echo "cuda: ${device}"
echo "testnumber: ${testnumber}"
echo "winmode: ${winmode}"
echo "winsize: ${winsize}"
echo "winZipSize: ${winZipSize}"
echo "winstride: ${winstride}"
echo "win use NAM: ${winuseConvRes}"
echo "usewinmem: ${usewinmem}"
echo "alignmode：${alignMode}"
echo "useNorm：${useNorm}"

# ===============================================================================================================

### train on the MVTec AD dataset
python train.py --dataset mvtec --train_data_path /data/name/database/MVTEC \
--save_path ./exps/visa/svit_large_14_518/fewshot/${model_name} --config_path ./open_clip/model_configs/ViT-L-14-336.json --model ViT-L-14-336 \
--few_shot_features 6 12 18 24 --pretrained openai --image_size 518  --batch_size 8 --aug_rate 0.3 --print_freq 1 \
--useAdapter ${useadapter} --useAbs ${useabs} --useFirstBatchNorm ${useFirstBatchNorm} --useConvRes ${useConvRes} --useSENet ${useSENet} \
--useZeroCls ${useZeroCls} --ResFusionMode ${ResFusionMode} --RRmode fewshot \
--epoch ${onmvtec_epochs} --save_freq 1 --device ${device} --alignMode ${alignMode} --useNorm ${useNorm}


### train on the VisA dataset
python train.py --dataset visa --train_data_path /data/name/database/VISA \
--save_path ./exps/mvtec/svit_large_14_518/fewshot/${model_name} --config_path ./open_clip/model_configs/ViT-L-14-336.json --model ViT-L-14-336 \
--few_shot_features 6 12 18 24 --pretrained openai --image_size 518  --batch_size 8 --print_freq 1 \
--useAdapter ${useadapter} --useAbs ${useabs} --useFirstBatchNorm ${useFirstBatchNorm} --useConvRes ${useConvRes} --useSENet ${useSENet} \
--useZeroCls ${useZeroCls} --ResFusionMode ${ResFusionMode} --RRmode fewshot \
--epoch ${onvisa_epochs} --save_freq 1 --device ${device} --use_crop ${usecrop} --alignMode ${alignMode} --useNorm ${useNorm}


### Test ===========================================================================================================

### test on the VisA dataset
 python test.py --mode few_shot --dataset visa \
 --data_path /data/name/database/VISA --save_path ./results/visa/few_shot/${model_name}/s${KSHOTS}shot/seed${seed} \
 --rrcount_path ./results/visa/few_shot/${model_name}/s${KSHOTS}shot/seed${seed}/rrcount.txt \
 --coscount_path ./results/visa/few_shot/${model_name}/s${KSHOTS}shot/seed${seed}/coscount.txt \
 --config_path ./open_clip/model_configs/ViT-L-14-336.json \
 --checkpoint_path ./exps/visa/svit_large_14_518/fewshot/${model_name}/epoch_${testvisa_epochs}.pth \
 --model ViT-L-14-336 --features_list 6 12 18 24 --few_shot_features 6 12 18 24 \
 --useAdapter ${useadapter} --useAbs ${useabs} --useFirstBatchNorm ${useFirstBatchNorm} --useConvRes ${useConvRes} --useSENet ${useSENet} \
 --useZeroCls ${useZeroCls} --ResFusionMode ${ResFusionMode} --RRmode ${testMode} --alignMode ${alignMode} --useNorm ${useNorm} \
 --winmode ${winmode} --winsize ${winsize} --winZipSize ${winZipSize} --winstride ${winstride} --winuseConvRes ${winuseConvRes} --usewinmem ${usewinmem} --usewinimgencoder ${usewinimgencoder} --winwight ${winwight} \
 --pretrained openai --image_size 518 --calImage_size 240 --k_shot ${KSHOTS} --seed ${seed} --device ${device} --saveimg ${testsaveimg} --count_res ${count_res} --useImgIdx ${useimgidx}


# ### test on the MVTec AD dataset
 python test.py --mode few_shot --dataset mvtec \
 --data_path /data/name/database/MVTEC --save_path ./results/mvtec/few_shot/${model_name}/s${KSHOTS}shot/seed${seed} \
 --rrcount_path ./results/mvtec/few_shot/${model_name}/s${KSHOTS}shot/seed${seed}/rrcount.txt \
 --coscount_path ./results/mvtec/few_shot/${model_name}/s${KSHOTS}shot/seed${seed}/coscount.txt \
 --config_path ./open_clip/model_configs/ViT-L-14-336.json \
 --checkpoint_path ./exps/mvtec/svit_large_14_518/fewshot/${model_name}/epoch_${testmvtec_epochs}.pth \
 --model ViT-L-14-336 --features_list 6 12 18 24 --few_shot_features 6 12 18 24 \
 --useAdapter ${useadapter} --useAbs ${useabs} --useFirstBatchNorm ${useFirstBatchNorm} --useConvRes ${useConvRes} --useSENet ${useSENet} \
 --useZeroCls ${useZeroCls} --ResFusionMode ${ResFusionMode} --RRmode ${testMode} --alignMode ${alignMode} --useNorm ${useNorm} \
 --winmode ${winmode} --winsize ${winsize} --winZipSize ${winZipSize} --winstride ${winstride} --winuseConvRes ${winuseConvRes} --usewinmem ${usewinmem} --usewinimgencoder ${usewinimgencoder} --winwight ${winwight} \
 --pretrained openai --image_size 518 --calImage_size 240 --k_shot ${KSHOTS} --seed ${seed} --device ${device} --saveimg ${testsaveimg} --count_res ${count_res} --useImgIdx ${useimgidx}