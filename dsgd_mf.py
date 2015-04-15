import sys
import random
import math
from pyspark import SparkContext, SparkConf
from numpy import *

#global command line args
num_factors= None 
num_workers = None
num_iterations = None
beta_value = None
lambda_value = None
inputV_filepath = None
outputW_filepath = None
outputH_filepath = None

#global variables
users_per_w_block = None
movies_per_h_block = None
pattern = None

def transform_V(entry, users_per_w_block, movies_per_h_block, pattern_br):
	row = entry[0] / users_per_w_block
	col = entry[1] / movies_per_h_block

	stratum_id = pattern_br[row][col]

	return (row, (entry, stratum_id))
#end def

def sgd_func(entry):
	node_id = entry[0]

	V = list(entry[1][0])	#((uid, movid, rating), stratum)
	W = list(entry[1][1])	#(uid, [])
	H = list(entry[1][2])	#(movid, [])

	#W and H as dicts
	W_map = {uid: row for (uid,row) in W}
	H_map = {mov_id: col for (mov_id,col) in H}

	num_updates = 0

	for entry in V:
		epsilon = pow((1000 + num_updates + total_updates.value),-beta_br.value)

		#read entry from V
		uid = entry[0][0]
		mov_id = entry[0][1]
		rating = entry[0][2]

		W_entry = array(W_map[uid])
		H_entry = array(H_map[mov_id])

		#gradient descent
		#n_W = len(filter(None, W_map[uid]))
		W_new = list(W_entry - (epsilon*(((-2*(rating - dot(W_entry, H_entry)))*H_entry) + (((2*lambda_br.value)/n_W.value[uid])*W_entry))))

		#n_H = len(filter(None, H_map[uid]))
		H_new = list(H_entry - (epsilon*(((-2*(rating - dot(W_entry, H_entry)))*W_entry) + (((2*lambda_br.value)/n_H.value[mov_id])*H_entry))))

		#update W and H
		W_map[uid] = W_new
		H_map[mov_id] = H_new

		num_updates += 1
	#end for

	#add no. of iterations to accumulator
	last_iter_total.add(num_updates)

	#return new W and H lists
	return ([(node_id,(uid, W_list)) for uid,W_list in W_map.items()], [(node_id,(mov_id, H_list)) for mov_id,H_list in H_map.items()])
#end def

def command_line_args_check(args):
	if len(args) != 9:
		sys.stderr.write("""spark-submit dsgd_mf.py <num_factors> <num_workers> <num_iterations> <beta_value> <lambda_value> \\
        <inputV_filepath> <outputW_filepath> <outputH_filepath>\n""")
		sys.exit(1)
	#end if
#end def

def parse_command_line_args(args):
	global num_factors
	global num_workers
	global num_iterations
	global beta_value
	global lambda_value
	global inputV_filepath
	global outputW_filepath
	global outputH_filepath

	num_factors = int(args[1])
	num_workers = int(args[2])
	num_iterations = int(args[3])
	beta_value = float(args[4])
	lambda_value = float(args[5])
	inputV_filepath = args[6]
	outputW_filepath = args[7]
	outputH_filepath = args[8]
#end if

if __name__ == "__main__":

	#check for command line arguments
	command_line_args_check(sys.argv)

	#parse command line arguments
	parse_command_line_args(sys.argv)

	#create the spark context object
	conf = SparkConf().setAppName("dsgd").setMaster("local")
	sc = SparkContext(conf=conf)

	#read input file into an rdd
	inputV_rdd = sc.textFile(inputV_filepath, num_workers).map(lambda line: line.split(',')).map(lambda entry: (int(entry[0]), int(entry[1]), int(entry[2])))
	
	#number of users
	num_users = inputV_rdd.max(lambda entry: entry[0])[0]

	#number of movies
	num_movies = inputV_rdd.max(lambda entry: entry[1])[1]

	#get counts for partial derivatives and broadcast
	n_W = [0 for i in range(num_users+1)]
	n_H = [0 for i in range(num_movies+1)]

	v_file = open(inputV_filepath)
	for line in v_file:
		line = line.strip().split(',')
		n_W[int(line[0])] += 1
		n_H[int(line[1])] += 1
	#end for
	v_file.close()

	n_W = sc.broadcast(n_W)
	n_H = sc.broadcast(n_H)

	#construct stratum pattern
	pattern = [[0 for x in range(num_workers)] for x in range(num_workers)]
	
	for i in range(num_workers):
		for j in range(i,num_workers):
			pattern[i][j] = j-i
		#end for
	#end for

	for i in range(1,num_workers):
		for j in range(i):
			pattern[i][j] = num_workers-i+j
		#end for
	#end for

	#calculate block dimensions and broadcast required values
	users_per_w_block = sc.broadcast(int(math.ceil((float(num_users+1)/num_workers))))
	movies_per_h_block = sc.broadcast(int(math.ceil((float(num_movies+1)/num_workers))))
	pattern_br = sc.broadcast(pattern)

	#convert V to new form
	V_rdd = inputV_rdd.map(lambda entry: transform_V(entry, users_per_w_block.value, movies_per_h_block.value, pattern_br.value))

	#partition V
	V_rdd = V_rdd.partitionBy(num_workers, lambda key: key).persist()

	#construct the W array
	W_rdd = sc.parallelize(range(num_users+1))
	W_rdd = W_rdd.map(lambda x: (x, [random.uniform(0, 5) for _ in range(0, num_factors)])).keyBy(lambda entry: entry[0]/users_per_w_block.value)
	W_rdd = W_rdd.partitionBy(num_workers, lambda key: key).persist()

	#construct the H array
	H_rdd = sc.parallelize(range(num_movies+1))
	H_rdd = H_rdd.map(lambda x: (x, [random.uniform(0, 5) for _ in range(0, num_factors)])).keyBy(lambda entry: entry[0]/movies_per_h_block.value)
	#H_rdd = H_rdd.partitionBy(num_workers, partition_h).persist()

	#broadcast beta and lambda
	beta_br = sc.broadcast(beta_value)
	lambda_br = sc.broadcast(lambda_value)

	total_updates = sc.broadcast(0)
	curr_stratum = sc.broadcast(0)
	last_iter_total = sc.accumulator(0)

	#SGD begins
	for iter in range(num_iterations):
		#filter current stratum data
		stratum_V_rdd = V_rdd.filter(lambda entry: entry[1][1]==curr_stratum.value)

		#partition H
		H_rdd = H_rdd.map(lambda entry: (pattern_br.value[curr_stratum.value][(entry[1][0]/movies_per_h_block.value)],entry[1]))
		H_rdd = H_rdd.partitionBy(num_workers, lambda key: key).persist()

		#group V, W and H into a stratum
		stratum_rdd = stratum_V_rdd.groupWith(W_rdd,H_rdd).partitionBy(num_workers, lambda key: key).persist()

		#parallel SGD on strata
		stratum_rdd = stratum_rdd.map(sgd_func, True)
		updated_WH = stratum_rdd.collect()

		W_list = []
		H_list = []

		#merge W and H from all partitions
		for W,H in updated_WH:
			W_list.extend(W)
			H_list.extend(H)
		#end for

		#replace W_rdd and H_rdd
		W_rdd = sc.parallelize(W_list)
		H_rdd = sc.parallelize(H_list)

		#update broadcasts
		prev_stratum = curr_stratum.value
		curr_stratum.unpersist()
		curr_stratum = sc.broadcast((prev_stratum+1)%num_workers)

		prev_total = total_updates.value
		total_updates.unpersist()
		total_updates = sc.broadcast(prev_total+last_iter_total.value)

		#clear accumulator
		last_iter_total = sc.accumulator(0)
	#end for

	#output W and H values to file
	W = W_rdd.collect()
	H = H_rdd.collect()

	w_file = open(outputW_filepath,'w')
	h_file = open(outputH_filepath, 'w')

	W = [(user_id, w_row) for (node_id, (user_id, w_row)) in W]
	W.sort(key=lambda tup: tup[0])

	H = [(mov_id, h_col) for (node_id, (mov_id, h_col)) in H]
	H.sort(key=lambda tup: tup[0])

	for entry in W:
		w_file.write(','.join(str(item) for item in entry[1]) + '\n')
	#end for
	w_file.close()

	for i in range(num_factors):
		out_list = [h_col[i] for (mov_id, h_col) in H]
		h_file.write(','.join(str(item) for item in out_list) + '\n')
	#end for
	h_file.close()

#end main