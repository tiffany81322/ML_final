import numpy as np
import pickle

sum_dict1 = np.zeros(300)
with open( 'glove.840B.300d.txt' ) as f :
	vector_list = f.readlines()
	dict1 = {}
	for i in range(len(vector_list)):
		tmp = vector_list[i].split()
		key = tmp[0]
		value = np.float32(tmp[1:])
		dict1[key] = value
		sum_dict1 = sum_dict1+value
f.close()
av_dict1 = sum_dict1/len(vector_list)
#print 1
#file1 = open('wordtovector.dat','w')
#print 2
#pickle.dump(dict1,file1)
#print 3
#file1.close()
print 4

file1 = open('av.dat','w')
pickle.dump(av_dict1,file1)
file1.close()
