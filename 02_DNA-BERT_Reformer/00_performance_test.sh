### test run
# outdir=./M0_722/

# CUDA_VISIBLE_DEVICES=1 /maps/projects/renlab/people/nkz325/Thesis/bin/python 00_performance_test_726.py \
#     --h5file ./data/0713_full_combined_data.h5 \
#     --model_path ${outdir}/model_epoch4.bin \
#     --tokenizer_dir ${outdir} \
#     --device cuda:0 \
#     --output_csv ${outdir}/test_results.csv \
    
#     # --output_dir ${outdir} \
#     # --roc_png ${outdir}/test_roc.png \
#     # --output_csv ${outdir}/test_results.csv \
#     # --roc_png ${outdir}/test_roc.png \
#     # --device 0 \
#     # --device cuda:0 \
#     # --batch_size 256

#!/bin/bash

# 设置变量
H5FILE="./data/0713_full_combined_data.h5"
MODELPATH="./M0_722/model_epoch4.bin"
TOKENIZERDIR="./M0_722"
OUTPUTDIR="./M0_722/evaluation_outputs"
DEVICE="cuda:0"
BATCHSIZE=256

# 创建输出目录（可选）
mkdir -p ${OUTPUTDIR}

# 运行评估脚本
CUDA_VISIBLE_DEVICES=1 /maps/projects/renlab/people/nkz325/Thesis/bin/python 00_performance_test_726.py \
  --h5file ${H5FILE} \
  --model_path ${MODELPATH} \
  --tokenizer_dir ${TOKENIZERDIR} \
  --output_dir ${OUTPUTDIR} \
  --device ${DEVICE} \
  --batch_size ${BATCHSIZE}

echo "Evaluation complete. Results saved in ${OUTPUTDIR}"


