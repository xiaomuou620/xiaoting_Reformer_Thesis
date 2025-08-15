outdir=./M0_722_e10/
mkdir ${outdir}

/maps/projects/renlab/people/nkz325/Thesis/bin/python train_reformer_bc_722.py \
	--outdir ${outdir} \
	--h5file ./data/0713_full_combined_data.h5 \
	--lr 2e-05 \
	--batch-size 64 \
	--epochs 10 \
  --device 0 1 2 3 \