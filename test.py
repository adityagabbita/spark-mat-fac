import sys
import random
from pyspark import SparkContext, SparkConf

if __name__ == "__main__":

	#create the spark context object
	conf = SparkConf().setAppName("testApp").setMaster("local")
	sc = SparkContext(conf=conf)

	a = []
	for i in range(3):
		a.append((i,[random.uniform(0, 5) for _ in range(0, 4)]))
	print a

	b = [(1, "a"), (2, "b")]
	c = [(10,"aa"), (20, "bb")]

	rdd1 = sc.parallelize(b)
	rdd2 = sc.parallelize(c)
	rdd = rdd1.cartesian(rdd2)
	print rdd.collect()
