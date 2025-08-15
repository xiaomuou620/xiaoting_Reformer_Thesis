outdir=./plant-dnabert-BPE_plant_fullData_e6_722/
mkdir ${outdir}

/maps/projects/renlab/people/nkz325/Thesis/bin/python train_reformer_bc_plants_722.py \
	--outdir ${outdir} \
	--model_name zhangtaolab/plant-dnabert-BPE \
	--h5file ./data/0713_full_combined_data.h5 \
	--lr 2e-05 \
	--batch-size 64 \
	--epochs 6 \
  --device 0 1 2 \
