import re
import pickle
import numpy
import collections
from keras.models import Sequential, slice_X
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import recurrent
from keras.optimizers import SGD
import numpy as np
import theano.tensor as T
import theano
import math

'''
ftest = open('VGG_19_fc7.dat', 'w')
for i in xrange(0, len(img_feature)):
	tmp = img_feature[i]
	ftest.write( tmp[6:] )
ftest.close()
'''
IMG_DIM = 4096

WORD_VEC_LEN = 300

HIDDEN_SIZE = 1024



# ======================= load data =================================
print 'load choice...'
numChoice = 5
count = 0
with open('choice_train.txt') as f:
	f_choice = f.readlines()
f.close()
choice_train = []
for i in xrange(0,len(f_choice)):
	tmp_arr = np.zeros((numChoice, WORD_VEC_LEN))
	tmp_str = f_choice[i].split()
	for j in xrange(0, numChoice):
		tmp_arr[j][:] = np.float32( tmp_str[j*WORD_VEC_LEN : (j+1)*WORD_VEC_LEN] )
	choice_train.append(tmp_arr)
del f_choice, tmp_arr, tmp_str

print 'load VGG feature...'
with open('VGG_19_fc7.dat', 'r') as f:
	img_line = f.readlines()
f.close()
img_feature_id = []
img_feature = []
for i in xrange(0, len(img_line)):
#	print 'i = ' + str(i)
	img_tmp = img_line[i].split()
	img_feature_id.append( img_tmp[0] )
	img_feature.append( np.float32(img_tmp[1:]) )
del img_line, img_tmp

print 'load x, y train...'
X_train = np.loadtxt('question_train.txt')
y_train = np.loadtxt('answer_train.txt')


print 'load image id...'
f = open('train_img_que_id.dat', 'r')
f_img_id = f.readlines()
f.close()
img_que_id = []
for line in xrange(0, len(f_img_id)):
	img_que_id.append(f_img_id[line])
del f_img_id



#=====================load solution===========================
def create_sol_list( filename ) :
	with open( filename,'r' ) as f :
		header = f.readline()
		file = f.readlines()

	solution_list = []
	for line in file :
		pos = line.rindex( '\t' )
		solution_list.append( line[pos+1] )

	return solution_list
#model = Sequential()

#model.add(recurrent.LSTM(HIDDEN_SIZE, input_shape=(None,WORD_VEC_LEN ),init='uniform',activation='tanh', inner_activation='tanh',return_sequences=True))
#model.add(Dropout(0.5))
#model.add( Dense(HIDDEN_SIZE,,activation='tanh') )
#model.add(Dropout(0.5))


#model.compile(loss='mse', optimizer='adam')
print 'load sol...'
sol_list = create_sol_list( 'final_project_pack/answer.train_sol' )



model = Sequential()

model.add(Dense(2*HIDDEN_SIZE, input_dim=WORD_VEC_LEN+IMG_DIM, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(HIDDEN_SIZE, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2*HIDDEN_SIZE, init='uniform'))
model.add(Dropout(0.5))
model.add(Dense(WORD_VEC_LEN+IMG_DIM, init='uniform'))
model.add(Activation('softmax'))

def cosine_distance(y_true,y_pred) :
	def _squared_magnitude(x):
		return T.sum( T.sqr(x) )

	return  1 - ( T.sum(y_true*y_pred)/( T.sqrt( _squared_magnitude(y_true) )*T.sqrt( _squared_magnitude(y_pred) ) )  )[()]

y1 = T.matrix('float32')
y2 = T.matrix('float32')

#test= theano.function(
#	inputs = [y1,y2],
#	outputs = cosine_distance(y1,y2)
#	)


sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')



# =================== iteration for input data =======================

splitNum = 5
len_of_data = X_train.shape[0]
split_len = int(math.floor(X_train.shape[0] / splitNum))
for i in xrange(0, splitNum):
	print 'train split #' + str(i)
	if i != splitNum:
		X_train_tmp = np.zeros( (split_len, IMG_DIM+WORD_VEC_LEN) )
		y_train_tmp = np.zeros( (split_len, IMG_DIM+WORD_VEC_LEN) )
		for j in xrange(0, split_len):
			ind_tmp = img_feature_id.index(img_que_id[i*split_len+j].split()[0])
			X_train_tmp[j][0:IMG_DIM] = img_feature[ind_tmp]
			y_train_tmp[j][0:IMG_DIM] = img_feature[ind_tmp]
			X_train_tmp[j][IMG_DIM:] = X_train[i*split_len+j][:]
			y_train_tmp[j][IMG_DIM:] = y_train[i*split_len+j][:]

		model.fit(X_train_tmp, y_train_tmp, nb_epoch=100, batch_size=128,verbose = 2)


	else:
		X_train_tmp = np.zeros( (len_of_data - i*split_len, IMG_DIM+WORD_VEC_LEN) )
		y_train_tmp = np.zeros( (len_of_data - i*split_len, WORD_VEC_LEN) )
		for j in xrange(0, len_of_data - i*split_len):
			ind_tmp = img_feature_id.index(img_que_id[i*split_len+j].split()[0])
			X_train_tmp[j][0:IMG_DIM] = img_feature[ind_tmp]
			y_train_tmp[j][0:IMG_DIM] = img_feature[ind_tmp]
			X_train_tmp[j][IMG_DIM:] = X_train[i*split_len+j][:]
			y_train_tmp[j][IMG_DIM:] = y_train[i*split_len+j][:]

		model.fit(X_train_tmp, y_train_tmp, nb_epoch=100, batch_size=128,verbose = 2)

del X_train_tmp, y_train_tmp
#model.fit(X_train, y_train, nb_epoch=20, batch_size=16,verbose = 2)

# ====================================================================

#score = model.evaluate(X_test, y_test, batch_size=16)



# =================== theano test function =========================
guess_vec = T.vector( 'float32' )
choice_arr = T.matrix( 'float32' )

ele_mul = choice_arr * guess_vec

len_g = T.sqrt( T.sum( T.sqr( guess_vec ) ) )

len_c = T.sqrt( T.sum( T.sqr( choice_arr ),axis = -1 ) )

len_u = len_g*len_c

inner_prod = T.sum( ele_mul,axis = -1 )

cosine_vec = inner_prod / len_u

test_func = theano.function(
	inputs = [guess_vec,choice_arr],
	outputs = cosine_vec
	)


# ========================== testing =================================
correct_count = 0
A_to_E = ['A','B','C','D','E']
splitNum = 50
len_of_data = X_train.shape[0]
split_len = int(math.floor(X_train.shape[0] / splitNum))
for i in xrange(0, splitNum):
	print 'test split #' + str(i)
	choice_train_tmp = []
	if i != splitNum:
		X_train_tmp = np.zeros( (split_len, IMG_DIM+WORD_VEC_LEN) )		
		for j in xrange(0, split_len):
			choice_tmp = np.zeros( (5, IMG_DIM+WORD_VEC_LEN) )
			ind_tmp = img_feature_id.index(img_que_id[i*split_len+j].split()[0])
			X_train_tmp[j][0:IMG_DIM] = img_feature[ind_tmp]
			choice_tmp[:, 0:IMG_DIM] = np.array([img_feature[ind_tmp],]*5)
			X_train_tmp[j][IMG_DIM:] = X_train[i*split_len+j][:]
			choice_tmp[:, IMG_DIM:] = choice_train[i*split_len+j]
			choice_train_tmp.append(choice_tmp)
		

	else:
		X_train_tmp = np.zeros( (len_of_data - i*split_len, IMG_DIM+WORD_VEC_LEN) )
		for j in xrange(0, len_of_data - i*split_len):
			choice_tmp = np.zeros( (5, IMG_DIM+WORD_VEC_LEN) )
			ind_tmp = img_feature_id.index(img_que_id[i*split_len+j].split()[0])
			X_train_tmp[j][0:IMG_DIM] = img_feature[ind_tmp]
			choice_tmp[:, 0:IMG_DIM] = np.array([img_feature[ind_tmp],]*5)
			X_train_tmp[j][IMG_DIM:] = X_train[i*split_len+j][:]
			choice_tmp[:, IMG_DIM:] = choice_train[i*split_len+j]
			choice_train_tmp.append(choice_tmp)
		

	
	guess = model.predict(X_train_tmp)


	
	for i in xrange( len(guess) ) :
		max_idx = numpy.argmax( test_func(guess[i].astype('float32'),choice_train_tmp[i].astype('float32')) )
		if sol_list[i] == sol_list[max_idx] :
			correct_count = correct_count + 1


print 'correct count = ' + str(correct_count)



