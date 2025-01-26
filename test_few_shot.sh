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

model_name=${MODEL_NAME}                                                   # model name
useadapter=0                                                               # use adapter?
useSENet=0
useabs=1                                                                   # use diff abs?
useFirstBatchNorm=1                                                        #
useConvRes=${USE_NAM}                                                      # use NAM?
usecrop=1                                                                  # visa crop?
useZeroCls=0                                                               # use zero cls?
ResFusionMode=nan                                                          # zero res fusion mode
testMode=fewshot                                                           # test mode: zeroshot fewshot

winmode=${WIN_MODE}                                                        # nan mul onlywin
winuseConvRes=0
winsize=${WIN_SIZE}
winZipSize=${WIN_SIZE}
winstride=${WIN_STRIDE}
usewinmem=1
usewinimgencoder=1
winwight=${WIN_WEIGHT}

device=${DEVICE}                                                           # cuda
testnumber=5                                                               #
epochs=9
seed=${SEED}
kshot=${KSHOTS}
count_res=0                                                                #

alignMode=mulmax                                                           # diff2 mulmax
useNorm=1

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

### Test ===========================================================================================================

### test on the VisA dataset
python test.py --mode few_shot --dataset visa \
--data_path /data/name/database/VISA --save_path ./results/visa/few_shot/${model_name}_winmode${winmode}/s${kshot}shot/seed${seed} \
--rrcount_path ./results/visa/few_shot/${model_name}/s${kshot}shot/seed${seed}/rrcount.txt \
--coscount_path ./results/visa/few_shot/${model_name}/s${kshot}shot/seed${seed}/coscount.txt \
--config_path ./open_clip/model_configs/ViT-L-14-336.json \
--checkpoint_path ./exps/visa/svit_large_14_518/fewshot/${model_name}/epoch_${testnumber}.pth \
--model ViT-L-14-336 --features_list 6 12 18 24 --few_shot_features 6 12 18 24 \
--useAdapter ${useadapter} --useAbs ${useabs} --useFirstBatchNorm ${useFirstBatchNorm} --useConvRes ${useConvRes} --useSENet ${useSENet} \
--useZeroCls ${useZeroCls} --ResFusionMode ${ResFusionMode} --RRmode ${testMode} --alignMode ${alignMode} --useNorm ${useNorm} \
--winmode ${winmode} --winsize ${winsize} --winZipSize ${winZipSize} --winstride ${winstride} --winuseConvRes ${winuseConvRes} --usewinmem ${usewinmem} --usewinimgencoder ${usewinimgencoder} --winwight ${winwight} \
--pretrained openai --image_size 518 --calImage_size 240 --k_shot ${kshot} --seed ${seed} --device ${device} --count_res ${count_res}


### test on the MVTec AD dataset
python test.py --mode few_shot --dataset mvtec \
--data_path /data/name/database/MVTEC --save_path ./results/mvtec/few_shot/${model_name}_winmode${winmode}/s${kshot}shot/seed${seed} \
--rrcount_path ./results/mvtec/few_shot/${model_name}/s${kshot}shot/seed${seed}/rrcount.txt \
--coscount_path ./results/mvtec/few_shot/${model_name}/s${kshot}shot/seed${seed}/coscount.txt \
--config_path ./open_clip/model_configs/ViT-L-14-336.json \
--checkpoint_path ./exps/mvtec/svit_large_14_518/fewshot/${model_name}/epoch_${testnumber}.pth \
--model ViT-L-14-336 --features_list 6 12 18 24 --few_shot_features 6 12 18 24 \
--useAdapter ${useadapter} --useAbs ${useabs} --useFirstBatchNorm ${useFirstBatchNorm} --useConvRes ${useConvRes} --useSENet ${useSENet} \
--useZeroCls ${useZeroCls} --ResFusionMode ${ResFusionMode} --RRmode ${testMode} --alignMode ${alignMode} --useNorm ${useNorm} \
--winmode ${winmode} --winsize ${winsize} --winZipSize ${winZipSize} --winstride ${winstride} --winuseConvRes ${winuseConvRes} --usewinmem ${usewinmem} --usewinimgencoder ${usewinimgencoder} --winwight ${winwight} \
--pretrained openai --image_size 518 --calImage_size 240 --k_shot ${kshot} --seed ${seed} --device ${device} --count_res ${count_res}