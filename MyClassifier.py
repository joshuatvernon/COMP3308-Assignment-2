#!/usr/bin/python3

from operator import itemgetter

import sys
import math

def read_data(fname):
	lines = []
	with open(fname) as f:
		lines = f.readlines()
	return lines

def get_data(fname):
	lines = read_data(fname)

	data = []
	for line in lines:
		line = line.strip()
		data.append(line.split(","))

	for dt in data:
		for i in range(0,len(dt)):
			if i != 8:
				dt[i] = float(dt[i])

	return data

def Euclidean_d(training_dt,input_dt):
	sum = 0
	for i in range(0,len(input_dt)):
		sum = math.pow(training_dt[i]-input_dt[i],2)
	return math.sqrt(sum)

def KNN(k,training_dt,input_dt):
	ls = []
	
	for train_dt in training_dt:
		dis = Euclidean_d(train_dt,input_dt)
		ls.append({"distance":dis,"class":train_dt[8]})

	pound  = []
	for elmt in ls:
		if (len(pound) < k):
			pound.append(elmt)
			ls = sorted(pound,key=itemgetter("distance"),reverse=True)
		else:
			for el in pound:
				if elmt["distance"] < el["distance"]:
					el["distance"] = elmt["distance"]
					el["class"] = elmt["class"]

	factor = 0
	for elmt in pound:
		if elmt["class"] == "yes":
			factor = factor + 1
		elif elmt["class"] == "no":
			factor = factor - 1

	if factor >= 0:
		return "yes"
	else:
		return "no"

def BN(training_dt,input_dt):
	return 1

def get_algo(algo,training_dt,input_dt):
	result = []
	if algo == "NB":
		for in_dt in input_dt:
			result.append(BN(training_dt,in_dt))
	elif "NN" in algo:
		k = int(algo.strip("NN"))
		for in_dt in input_dt:
			result.append(KNN(k,training_dt,in_dt))
	else:
		return 1

	return result

def print_result(result):
	for rs in result:
		print (rs)

def main(argv):
	if (len(argv) != 3):
		return 1

	training_file = argv[0]
	input_file = argv[1]
	algo = argv[2]


	training_dt = get_data(training_file)
	input_dt = get_data(input_file)

	result = get_algo(algo,training_dt,input_dt)

	print_result(result)

if __name__ == "__main__":
   main(sys.argv[1:])