all:
	python main.py --model ${rl} --train 1 --load_num ${l_num} --train_num ${t_num}
test:
	python main.py --model  ${rl} --train 0 --watch 1 --watch_num ${num}