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
import inflect
import string
from stop_words import get_stop_words

stop_word = get_stop_words('english')
print 'load dictionary...'
with open('wordtovector.dat','r') as f :
	dict = pickle.load( f )

WORD_VEC_LEN = 300

HIDDEN_SIZE = 1024

replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation) )


if_e = inflect.engine()

num_word = if_e.number_to_words

#=====================load dict=================================
def load_dict(filename) :
  word_vec={}
  with open(filename,'rb') as f :
    header = f.readline()
    dict_size , layer_size = map( int,header.split() )
    binary_len = np.dtype( 'float32' ).itemsize*layer_size
    for line in xrange( dict_size ) :
      word = []
      while True :
        ch = f.read( 1 )
        if ch == ' ' :
          word = ''.join(word)
          break
        if ch != '\n' :
          word.append(ch)
      word_vec[word] = np.fromstring( f.read(binary_len),dtype='float32' )

  return word_vec

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
		ques = ques.translate( replace_punctuation ).split()
		vec_array = numpy.zeros(300)
		count=0
		for word in ques :
			tmp_word = word.lower()
			if tmp_word in stop_word:
				continue
			else:
				try :
					vec_array = vec_array + dict[ tmp_word ]
					count = count + 1
				except KeyError :
					if num_word( tmp_word ) != "zero" and num_word( tmp_word ) != "zeroth" :
						number_words = num_word( tmp_word )
						number_words = number_words.translate( replace_punctuation ).split()
						for number in number_words :
							if number != "and" :
								vec_array = vec_array + dict.get( tmp_word,numpy.zeros(300) )
								count = count + 1



		ques_vec_dict[ img_q_id[0],img_q_id[1] ] = vec_array

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
		ans = ans.translate( replace_punctuation ).split()

		vec_array = numpy.zeros(300)
		count = 0
		for word in ans :
			tmp_word = word.lower()
			if tmp_word in stop_word:
				continue
			else:
				try :
					vec_array = vec_array + dict[ tmp_word ]
					count = count+1
				except KeyError :
					if num_word( tmp_word ) != "zero" and num_word( tmp_word ) != "zeroth" :
						number_words = num_word( tmp_word )
						number_words = number_words.translate( replace_punctuation ).split()
						for number in number_words :
							if number != "and" :
								vec_array = vec_array + dict.get( tmp_word,numpy.zeros(300) )
								count = count + 1

		if  count ==0 :
			ans_vec_dict[ img_q_id[0],img_q_id[1] ] =numpy.zeros(300)
		else :
			ans_vec_dict[ img_q_id[0],img_q_id[1] ] = vec_array

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
				word_list = choice_line.translate( replace_punctuation ).split()
				vec_array = numpy.zeros(300)
				count = 0
				for word in word_list :
					tmp_word = word.lower()
					if tmp_word in stop_word:
						continue
					else:
						try :
							vec_array = vec_array + dict[ tmp_word ]
							count = count+1
						except KeyError :
							if num_word( tmp_word ) != "zero" and num_word( tmp_word ) != "zeroth" :
								number_words = num_word( tmp_word )
								number_words = number_words.translate( replace_punctuation ).split()
								for number in number_words :
									if number != "and" :
										vec_array = vec_array + dict.get( tmp_word,numpy.zeros(300) )
										count = count + 1

				if count == 0 :
					vec_list.append( 0.001*numpy.random.randn(300) )
				else :
					vec_list.append( vec_array )
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

#=================load annotation===============
def create_anno_arr( filename ) :
	with open( filename,'r' ) as f :
		header = f.readline()
		file = f.readlines()
	#type_list = [ ,'answer_type','\n' ]
	vec_array = numpy.zeros(300)
	for line in file :
		q_type_start = line.index('question_type')
		ans_type_start = line.index('"\tanswer_type')
		q_type_string = line[ q_type_start:ans_type_start ].replace( 'question_type:"','' )
		string_end = line.rindex('"')
		ans_type_string = line[ans_type_start:string_end].replace('"\tanswer_type:"','')
		words_q_type = line


#model = Sequential()

#model.add(recurrent.LSTM(HIDDEN_SIZE, input_shape=(None,WORD_VEC_LEN ),init='uniform',activation='tanh', inner_activation='tanh',return_sequences=True))
#model.add(Dropout(0.5))
#model.add( Dense(HIDDEN_SIZE,,activation='tanh') )
#model.add(Dropout(0.5))


#model.compile(loss='mse', optimizer='adam')
print 'load data...'
X_train = numpy.vstack( create_ques_data( 'final_project_pack/question.train' ).values() )
y_train = numpy.vstack( create_ans_data( 'final_project_pack/answer.train' ).values() )
choice_train = create_choice_arr( 'final_project_pack/choices.train' ).values()
sol_list = create_sol_list( 'final_project_pack/answer.train_sol' )
anno_arr = create_anno_arr( 'final_project_pack/annotation.train' )
print 'run keras...'
model = Sequential()

model.add(Dense(HIDDEN_SIZE, input_dim=WORD_VEC_LEN, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(HIDDEN_SIZE, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(HIDDEN_SIZE, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(WORD_VEC_LEN, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
'''
model.add(Dropout(0.2))
model.add(Dense(WORD_VEC_LEN, init='uniform'))
model.add(Activation('tanh'))
'''
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


sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
model.compile(loss="mean_squared_error", optimizer=sgd)

model.fit(X_train, y_train, nb_epoch=100, batch_size=48, verbose = 2)
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
print 'start testing...'
correct_count = 0
A_to_E = ['A','B','C','D','E']
for i in xrange( len(guess) ) :
	max_idx = numpy.argmax( test_func(guess[i].astype('float32'),choice_train[i].astype('float32')) )
	if sol_list[i] == sol_list[max_idx] :
		correct_count = correct_count + 1

print 'accuracy = ' + str(float(correct_count)/len(guess))
