#!/usr/bin/python3

from operator import itemgetter

import sys
import math
import csv

index_last_col = 5	# if testing CFS 5

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
			if i != index_last_col:
				dt[i] = float(dt[i])

	return data

def Euclidean_d(training_dt,input_dt):
	sum = 0
	for i in range(0,len(input_dt)):
		sum = sum + math.pow(training_dt[i]-input_dt[i],2)

	return math.sqrt(sum)

def KNN(k,training_dt,input_dt):
	ls = []

	for train_dt in training_dt:
		dis = Euclidean_d(train_dt,input_dt)
		ls.append({"distance":dis,"class":train_dt[index_last_col]})

	pound  = []
	for elmt in ls:
		if (len(pound) < k):
			pound.append(elmt)
			pound = sorted(pound,key=itemgetter("distance"),reverse=True)
		else:
			for el in pound:
				if elmt["distance"] < el["distance"]:
					el["distance"] = elmt["distance"]
					el["class"] = elmt["class"]
					break
			pound = sorted(pound,key=itemgetter("distance"),reverse=True)

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

def get_mean(data):
	sum = []
	for i in range(0,index_last_col):
		sum.append({"yes":0,"no":0})
	num_of_yes = 0
	for dt in data:
		if dt[index_last_col] == "yes":
			num_of_yes = num_of_yes + 1
			for i in range(0,len(dt)-1):
				sum[i]["yes"] = sum[i]["yes"] + dt[i]
		elif dt[index_last_col] == "no":
			for i in range(0,len(dt)-1):
				sum[i]["no"] = sum[i]["no"] + dt[i]

	mean = []
	for num in sum:
		mean_yes = num["yes"]/num_of_yes
		mean_no = num["no"]/(len(data)-num_of_yes)
		mean.append({"yes":mean_yes,"no":mean_no})

	mean.append(num_of_yes)

	return mean

def get_standard_deviation(training_dt,mean):
	sum = []
	for i in range(0,index_last_col):
		sum.append({"yes":0,"no":0})
	for dt in training_dt:
		for i in range(0,len(dt)-1):
			num = math.pow(dt[i] - mean[i][dt[index_last_col]],2)
			sum[i][dt[index_last_col]] = sum[i][dt[index_last_col]] + num

	standard_deviation = []
	for num in sum:
		standard_deviation_yes = math.sqrt(num["yes"]/(mean[-1]-1))
		standard_deviation_no = math.sqrt(num["no"]/(len(training_dt)-mean[-1]-1))
		standard_deviation.append({"yes":standard_deviation_yes,"no":standard_deviation_no})

	return standard_deviation

def probability_density_function(mean,standard_deviation,x):
	result = (1/(standard_deviation*math.sqrt(2*math.pi))) * math.exp(-math.pow(x-mean,2)/(2*math.pow(standard_deviation,2)))

	return result

def NB(training_dt,input_dt):
	mean = get_mean(training_dt)
	standard_deviation = get_standard_deviation(training_dt,mean)

	num_of_yes = mean[-1]

	result = []
	for dt in input_dt:
		probability_yes = []
		probability_no = []
		for i in range(0,len(dt)):
			probability_yes.append(probability_density_function(mean[i]["yes"],standard_deviation[i]["yes"],dt[i]))
			probability_no.append(probability_density_function(mean[i]["no"],standard_deviation[i]["no"],dt[i]))
		x_yes = num_of_yes/len(training_dt)
		x_no = (len(training_dt) - num_of_yes)/len(training_dt)
		for p in probability_yes:
			if p != 0:
				x_yes = x_yes * p
		for p in probability_no:
			if p != 0:
				x_no = x_no * p

		if x_no > x_yes:
			result.append("no")
		else:
			result.append("yes")

	return result

def get_algo(algo,training_dt,input_dt):
	result = []
	if algo == "NB":
		result = NB(training_dt,input_dt)
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

def folds(training_dt):
	ls_yes = []
	ls_no = []
	for dt in training_dt:
		if dt[index_last_col] == "yes":
			ls_yes.append(dt)
		elif dt[index_last_col] == "no":
			ls_no.append(dt)

	folds = []
	for i in range(0,10):
		folds.append([])

	len_yes = int(len(ls_yes)/10)
	rmd_yes = len(ls_yes)%10
	len_no = int(len(ls_no)/10)
	rmd_no = len(ls_no)%10

	for i in range(0,10):
		for j in range(0,len_yes):
			folds[i].append(ls_yes.pop(0))
		for j in range(0,len_no):
			folds[i].append(ls_no.pop(0))

	for i in range(0,10):
		if (len(ls_yes) > 0):
			folds[i].append(ls_yes.pop(0))
		if (len(ls_no) > 0):
			folds[i].append(ls_no.pop(0))


	with open('pima-folds.csv', 'w', newline='') as myfile:
		wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
		i = 1
		for fold in folds:
			wr.writerow(["fold"+str(i)])
			i = i + 1
			for row in fold:
				wr.writerow(row)

def fold_cross_validation(fname,algo):

	lines = []
	with open(fname) as f:
		lines = f.readlines()

	training_dt_ls = []
	test_dt_ls = []

	for i in range(1,11):
		training_data = []
		test_data = []
		test = False
		for ln in lines:
			ln = ln.strip()
			if "fold" in ln:
				ln = ln.strip("fold")
				if int(ln) == i:
					test = True
				else:
					test = False
			else:
				dt = []
				dt = ln.split(",")
				x = [float(i) for i in dt[0:-1]]
				x.append(dt[-1])

				if test:
					test_data.append(x)
				else:
					training_data.append(x)

		training_dt_ls.append(training_data)
		test_dt_ls.append(test_data)

	result_ls = []
	for i in range(0,10):
		result = get_algo(algo,training_dt_ls[i],[ls[0:-1] for ls in test_dt_ls[i]])
		result_ls.append(compare_ls([ls[-1] for ls in test_dt_ls[i]],result))

	sum = 0
	for i in range(0,len(result_ls)):
		sum = sum + result_ls[i]

	print(float(sum/len(result_ls)))

def compare_ls(ls1,ls2):
	if len(ls1) != len(ls2):
		print("something is wrong")

	count = 0

	for i in range(0,len(ls1)):
		if ls1[i] == ls2[i]:
			count = count + 1

	result = float(count/len(ls1))

	return result


def main(argv):
	if (len(argv) == 2):
		return fold_cross_validation(argv[0],argv[1])
	elif (len(argv) != 3):
		return 1

	training_file = argv[0]
	input_file = argv[1]
	algo = argv[2]


	training_dt = get_data(training_file)
	input_dt = get_data(input_file)

	#folds(training_dt)

	result = get_algo(algo,training_dt,input_dt)

	print_result(result)

if __name__ == "__main__":
   main(sys.argv[1:])