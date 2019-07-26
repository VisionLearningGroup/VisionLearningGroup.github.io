import torch

def euclidean(x1,x2):
	return ((x1-x2)**2).sum().sqrt()

def k_moment(output_s1, output_s2, output_s3, output_s4, output_t, k):
	output_s1 = (output_s1**k).mean(0)
	output_s2 = (output_s2**k).mean(0)
	output_s3 = (output_s3**k).mean(0)
	output_t = (output_t**k).mean(0)
	return  euclidean(output_s1, output_t) + euclidean(output_s2, output_t) + euclidean(output_s3, output_t)+ \
		euclidean(output_s1, output_s2) + euclidean(output_s2, output_s3) + euclidean(output_s3, output_s1) +\
		euclidean(output_s4, output_s1) + euclidean(output_s4, output_s2) + euclidean(output_s4, output_s2) + \
		euclidean(output_s4, output_t)

def msda_regulizer(output_s1, output_s2, output_s3, output_s4, output_t, belta_moment):
	# print('s1:{}, s2:{}, s3:{}, s4:{}'.format(output_s1.shape, output_s2.shape, output_s3.shape, output_t.shape))
	s1_mean = output_s1.mean(0)
	s2_mean = output_s2.mean(0)
	s3_mean = output_s3.mean(0)
	t_mean = output_t.mean(0)
	output_s1 = output_s1 - s1_mean
	output_s2 = output_s2 - s2_mean
	output_s3 = output_s3 - s3_mean
	output_t = output_t - t_mean
	moment1 = euclidean(output_s1, output_t) + euclidean(output_s2, output_t) + euclidean(output_s3, output_t)+\
		euclidean(output_s1, output_s2) + euclidean(output_s2, output_s3) + euclidean(output_s3, output_s1) +\
		euclidean(output_s4, output_s1) + euclidean(output_s4, output_s2) + euclidean(output_s4, output_s2) + \
		euclidean(output_s4, output_t)
	reg_info = moment1
	#print(reg_info)
	for i in range(belta_moment-1):
		reg_info += k_moment(output_s1,output_s2,output_s3, output_s4, output_t,i+2)
	
	return reg_info/6
	#return euclidean(output_s1, output_t)

