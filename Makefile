train: 
	-rm ./summaries/reinforce_logdir/*
	python main.py

clean:
	rm ./summaries/reinforce_logdir/*

