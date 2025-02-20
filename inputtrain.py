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

with open('wordtovector.dat','r') as f :
	dict = pickle.load( f )

#with open('av.dat','r') as f :
	#av_dict = pickle.load( f )


WORD_VEC_LEN = 300

HIDDEN_SIZE = 1024

#============question.train=======================
def create_ques_data( filename ) :
	dict_key = dict.keys()
	with open( filename,'r' ) as f :
		header = f.readline()
		file = f.readlines()

	ques_vec_dict=collections.OrderedDict()
	for line in file :
		start = line.index('"')
		end = line.rindex('"')
		ques = line[start+1:end]
		img_q_id = line[:start].split('\t')
		ques = re.sub(r'[^\w]', ' ', ques).split()
		#vec_list=[]
		vec_array = numpy.zeros(300)
		count=0
		for word in ques :
			#vec_list.append( dict.get( word,numpy.random.randn(300) ) )
			#vec_array = vec_array + dict.get( word,numpy.random.randn(300) )
			vec_array = vec_array + dict.get( word,numpy.zeros(300) )
			count = count+1
		#ques_vec_dict[ img_q_id[0],img_q_id[1] ] = numpy.vstack( vec_list )
		if count == 0:
			ques_vec_dict[ img_q_id[0],img_q_id[1] ] = numpy.zeros(300)
		else:
			ques_vec_dict[ img_q_id[0],img_q_id[1] ] = vec_array/count

	return ques_vec_dict

#==================answer train=============================
def create_ans_data( filename ) :
	with open( filename,'r' ) as f :
		header = f.readline()
		file = f.readlines()

	ans_vec_dict = collections.OrderedDict()
	for line in file :
		start = line.index('"')
		end = line.rindex('"')
		ans = line[start+1:end]
		img_q_id = line[:start].split('\t')
		ans = re.sub(r'[^\w]', ' ', ans).split()
		#vec_list=[]
		vec_array = numpy.zeros(300)
		count = 0
		for word in ans :
			#vec_array = vec_array + dict.get( word,0.001*numpy.random.randn(300) )
			vec_array = vec_array + dict.get( word,0.001*numpy.zeros(300) )
			count = count+1
			#vec_list.append( dict.get( word,numpy.random.randn(300) ) )
		if  count ==0 :
			#ans_vec_dict[ img_q_id[0],img_q_id[1] ] =0.001*numpy.random.randn(300)
			ans_vec_dict[ img_q_id[0],img_q_id[1] ] =0.001*numpy.zeros(300)
		else :
			#ans_vec_dict[ img_q_id[0],img_q_id[1] ] = numpy.vstack( vec_list )
			ans_vec_dict[ img_q_id[0],img_q_id[1] ] = vec_array/count

	return ans_vec_dict

#====================choice.train=============================
def create_choice_arr( filename ) :
	with open( filename,'r' ) as f :
		header = f.readline()
		file = f.readlines()

		choice_vec_dict = collections.OrderedDict()
		choice_alp = ['(A)','(B)','(C)','(D)','(E)','\n']
		for line in file :
			vec_list = []
			A_index = line.index('(A)')
			img_q_id = line[:A_index ].split('\t')
			for choice_idx in xrange( 5 ) :
				start = line.index( choice_alp[ choice_idx ] )
				end = line.index( choice_alp[ choice_idx + 1 ] )
				choice_line = line[ start:end ]
				choice_line.replace( choice_alp[ choice_idx ],'' )
				word_list = re.sub(r'[^\w]', ' ', choice_line).split()
				vec_array = numpy.zeros(300)
				count = 0
				for word in word_list :
					#vec_array = vec_array + dict.get( word,0.001*numpy.random.randn(300) )
					vec_array = vec_array + dict.get( word,0.001*numpy.zeros(300) )
					count = count+1
				if count == 0 :
					#vec_list.append( 0.001*numpy.random.randn(300) )
					vec_list.append( 0.001*numpy.zeros(300) )
				else :
					vec_list.append( vec_array/count )
			choice_vec_dict[ img_q_id[0],img_q_id[1] ] = numpy.vstack( vec_list )

	return choice_vec_dict

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

X_train = numpy.vstack( create_ques_data( 'final_project_pack/question.train' ).values() )
y_train = numpy.vstack( create_ans_data( 'final_project_pack/answer.train' ).values() )
choice_train = create_choice_arr( 'final_project_pack/choices.train' ).values()
sol_list = create_sol_list( 'final_project_pack/answer.train_sol' )

model = Sequential()

model.add(Dense(2048, input_dim=WORD_VEC_LEN, init='uniform'))
model.add(Activation('tanh'))
#model.add(Dropout(0.5))
model.add(Dense(1024, init='uniform'))
model.add(Activation('tanh'))
#model.add(Dropout(0.5))
model.add(Dense(2048, init='uniform'))
#model.add(Dropout(0.5))
model.add(Dense(WORD_VEC_LEN, init='uniform'))
model.add(Activation('tanh'))

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


sgd = SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=cosine_distance, optimizer=sgd)

model.fit(X_train, y_train, nb_epoch=20, batch_size=128,verbose = 2)
#score = model.evaluate(X_test, y_test, batch_size=16)
guess = model.predict(X_train)

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

correct_count = 0
A_to_E = ['A','B','C','D','E']
for i in xrange( len(guess) ) :
	max_idx = numpy.argmax( test_func(guess[i].astype('float32'),choice_train[i].astype('float32')) )
	if sol_list[i] == sol_list[max_idx] :
		correct_count = correct_count + 1
print float(correct_count)/float(X_train.shape[0])
