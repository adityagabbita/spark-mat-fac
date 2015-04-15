import sys
import random
import math
from pyspark import SparkContext, SparkConf

#parse inputV file
def parse_v_file(entry):
	file_part_name = entry[0]
	text = entry[1].splitlines()

	#parse movie id
	movie_id = int(text[0].split(":")[0]) - 1
	
	ratings = []

	for line in text[1:]:
		line_entry = line.split(",")
		user_id = int(line_entry[0])
		user_rating = line_entry[1]
		ratings.append((user_id, movie_id, user_rating))
	#end for

	return ratings
#end def

def user_id_mapping(user_list):
	inc_id = 0
	id_map = {}

	#assign every unique user an auto-increment id
	for entry in user_list:
		id_map[entry[0]] = inc_id
		inc_id += 1
	#end for

	return id_map
#end def

#global variables
users_per_w_block = 0
movies_per_h_block = 0

def partition_w(user_id):
	return int(math.floor(float(user_id)/users_per_w_block))
#end def

def partition_h(movie_id):
	return int(math.floor(float(movie_id)/movies_per_h_block))
#end def

def filter_V(user_id,movie_id,i,pattern,users_per_w_block,movies_per_h_block):
	row = int(math.floor(float(user_id)/users_per_w_block))
	col = int(math.floor(float(movie_id)/movies_per_h_block))

	return (pattern[row][col] == i)
#end def

def l2_norm(m_rdd):
	return m_rdd.flatMap(lambda (x,y): [i ** 2 for i in y]).sum()
#end def

def loss_sum(v_rdd, w_rdd, h_rdd):
	print "a"	
#end def

if __name__ == "__main__":

	#check for command line arguments
	if(len(sys.argv) != 9):
		sys.stderr.write("""spark-submit dsgd_mf.py <num_factors> <num_workers> <num_iterations> <beta_value> <lambda_value> \\
        <inputV_filepath> <outputW_filepath> <outputH_filepath>\n""")
		sys.exit(1)
	#end if

	#parse command line arguments
	num_factors = int(sys.argv[1])
	num_workers = int(sys.argv[2])
	num_iterations = int(sys.argv[3])
	beta_value = float(sys.argv[4])
	lambda_value = float(sys.argv[5])
	inputV_filepath = sys.argv[6]
	outputW_filepath = sys.argv[7]
	outputH_filepath = sys.argv[8]

	#create the spark context object
	conf = SparkConf().setAppName("testApp").setMaster("local")
	sc = SparkContext(conf=conf)

	#read inputV file from directory path
	inputV_rdd = sc.wholeTextFiles(inputV_filepath)
	
	#number of movies
	num_movies = inputV_rdd.count()

	#parse data from input
	inputV_rdd = inputV_rdd.flatMap(parse_v_file)

	#list of (userid, count)
	user_list =  sorted(inputV_rdd.countByKey().items())
	#assign every user a new id map in (0,n)
	user_map = user_id_mapping(user_list)

	#number of users
	num_users = len(user_map)

	#change id of users
	inputV_rdd = inputV_rdd.map(lambda (x,y,z): (user_map[x],y,z)).persist()

	users_per_w_block = int(math.ceil((float(num_users)/num_workers)))
	movies_per_h_block = int(math.ceil((float(num_movies)/num_workers)))

	#construct the W array
	W_rdd = sc.parallelize(range(num_users))
	W_rdd = W_rdd.map(lambda x: (x, [random.uniform(0, 5) for _ in range(0, num_factors)]))
	W_rdd = W_rdd.partitionBy(num_workers, partition_w).persist()

	#construct the H array
	H_rdd = sc.parallelize(range(num_movies))
	H_rdd = H_rdd.map(lambda x: (x, [random.uniform(0, 5) for _ in range(0, num_factors)]))
	H_rdd = H_rdd.partitionBy(num_workers, partition_h).persist()	

	#construct pattern
	pattern = [[0 for x in range(num_workers)] for x in range(num_workers)]
	
	for i in range(num_workers):
		for j in range(i,num_workers):
			pattern[i][j] = j-1
		#end for
	#end for

	for i in range(1,num_workers):
		for j in range(i):
			pattern[i][j] = num_workers-i+j
		#end for
	#end for

	total_updates = 0

	#SGD begins
	for iter in range(num_iterations):
		for i in range(num_workers):
			#filter V data
			stratum_rdd = inputV_rdd.filter(lambda (x,y,z): filter_V(x,y,i,pattern,users_per_w_block,movies_per_h_block))
			
			iter_updates = stratum_rdd.count()
			epsilon = pow((100 + total_updates),beta_value)
			
			#partition startum into blocks
			stratum_rdd = stratum_rdd.partitionBy(num_workers, partition_w)
			
			#W_rdd = W_rdd.map(sgd_w(stratum_rdd, H_rdd, epsilon, lambda_value))
			#H_rdd = H_rdd.map(sgd_h(stratum_rdd, W_rdd, epsilon, lambda_value))
		#end for
	#end for

#end main
