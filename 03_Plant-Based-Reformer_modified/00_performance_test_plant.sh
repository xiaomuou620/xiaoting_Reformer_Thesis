### test run
# reformer modified
outdir=./save_reformer_bc_plant_fullData_e5/

/maps/projects/renlab/people/nkz325/Thesis/bin/python 00_performance_test_plant.py \
    --h5file ./data/0713_full_combined_data.h5 \
    --model_path ${outdir}/model_best.bin \
    --tokenizer_dir ${outdir} \
    --output_csv ${outdir}/test_results.csv \
    --roc_png ${outdir}/test_roc.png \
    --device cuda:0 \
