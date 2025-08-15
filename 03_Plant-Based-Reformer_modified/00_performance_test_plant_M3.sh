### test run
# reformer modified
outdir=./M4_722/


CUDA_VISIBLE_DEVICES=2 /maps/projects/renlab/people/nkz325/Thesis/bin/python 00_performance_test_plant_722.py \
    --h5file ./data/0713_full_combined_data.h5 \
    --model_path ${outdir}/model_epoch2.bin \
    --model_name zhangtaolab/plant-dnabert-BPE-conservation \
    --tokenizer_dir ${outdir} \
    --output_csv ${outdir}/test_results.csv \
    --roc_png ${outdir}/test_roc.png \
    --device cuda:0 \
