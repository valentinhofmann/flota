#!/bin/bash

for data in arxiv_cs_1e+02 arxiv_maths_1e+02 arxiv_physics_1e+02 arxiv_cs_1e+03 arxiv_maths_1e+03 arxiv_physics_1e+03
do
	python3.8 -u main.py \
	--batch_size 64 \
	--n_epochs 20 \
	--data $data \
	--base \
	--noise test \
	--model "$1" \
	--lr "$2" \
	--device "$3"
done

for k in 1 2 3 4
do 
	for data in arxiv_cs_1e+02 arxiv_maths_1e+02 arxiv_physics_1e+02 arxiv_cs_1e+03 arxiv_maths_1e+03 arxiv_physics_1e+03
	do
		python3.8 -u main.py \
		--batch_size 64 \
		--n_epochs 20 \
		--data $data \
		--flota \
		--k $k \
		--noise test \
		--model "$1" \
		--lr "$2" \
		--device "$3"
	done
done
