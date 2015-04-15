all:
	/afs/cs.cmu.edu/project/bigML/spark-1.3.0-bin-hadoop2.4/bin/spark-submit --master local[*] ./dsgd_mf.py 10 10 50 0.8 1.0 ./input/ w.csv h.csv

test:
	/afs/cs.cmu.edu/project/bigML/spark-1.3.0-bin-hadoop2.4/bin/spark-submit ./test.py
