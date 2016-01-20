import re
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
import csv
import pickle

WORD_VEC_LEN = 300

HIDDEN_SIZE = 4096

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

dict = load_dict('GoogleNews-vectors-negative300.bin')

#with open('wordtovector.dat','r') as f :
	#dict = pickle.load(f)

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
			try :
				vec_array = vec_array + dict[ word ]
				count = count + 1
			except KeyError :
				if num_word( word ) != "zero" and num_word( word ) != "zeroth" :
					number_words = num_word( word )
					number_words = number_words.translate( replace_punctuation ).split()
					for number in number_words :
						if number != "and" :
							vec_array = vec_array + dict.get( word,numpy.zeros(300) )
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
			try :
				vec_array = vec_array + dict[ word ]
				count = count+1
			except KeyError :
				if num_word( word ) != "zero" and num_word( word ) != "zeroth" :
					number_words = num_word( word )
					number_words = number_words.translate( replace_punctuation ).split()
					for number in number_words :
						if number != "and" :
							vec_array = vec_array + dict.get( word,numpy.zeros(300) )
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
					try :
						vec_array = vec_array + dict[ word ]
						count = count+1
					except KeyError :
						if num_word( word ) != "zero" and num_word( word ) != "zeroth" :
							number_words = num_word( word )
							number_words = number_words.translate( replace_punctuation ).split()
							for number in number_words :
								if number != "and" :
									vec_array = vec_array + dict.get( word,numpy.zeros(300) )
									count = count + 1

				if count == 0 :
					vec_list.append( 0.001*numpy.random.randn(300) )
				else :
					vec_list.append( vec_array )
			choice_vec_dict[ img_q_id[0],img_q_id[1] ] = numpy.hstack( vec_list )

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

X_train = create_ques_data( 'final_project_pack/question.train' ).values()
X_test_dict = create_ques_data( 'final_project_pack/question.test' )
y_train =  create_ans_data( 'final_project_pack/answer.train' ).values()
choice_train = create_choice_arr( 'final_project_pack/choices.train' ).values()
choice_test_dict = create_choice_arr( 'final_project_pack/choices.test' )
sol_list = create_sol_list( 'final_project_pack/answer.train_sol' )
anno_arr = create_anno_arr( 'final_project_pack/annotation.train' )

NN_in=[]
for i in xrange( len( X_train ) ) :
	NN_in.append( numpy.hstack( [X_train[ i ],choice_train[ i ] ] ) )
NN_in = numpy.vstack( NN_in )

NN_test = []
test_ques_value = X_test_dict.values()
test_choice_value = choice_test_dict.values()
for i in xrange( len( X_test_dict ) ) :
	NN_test.append( numpy.hstack( [ test_ques_value[ i ],test_choice_value[ i ] ] ) )
NN_test = numpy.vstack( NN_test )

A_to_E = ['A','B','C','D','E']
sol_class = []
for sol in sol_list :
	tmp_array = numpy.zeros(5)
	tmp_array[ A_to_E.index(sol) ] = 1
	sol_class.append( tmp_array )
sol_class = numpy.vstack( sol_class )



model = Sequential()

model.add(Dense(HIDDEN_SIZE, input_dim=6*WORD_VEC_LEN, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(HIDDEN_SIZE, init='uniform'))
model.add(Activation('tanh'))
model.add(Dense(HIDDEN_SIZE, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(HIDDEN_SIZE, init='uniform'))
model.add(Activation('tanh'))
model.add(Dense(0.5*HIDDEN_SIZE, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(0.25*HIDDEN_SIZE, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(5, init='uniform'))
model.add(Activation('softmax'))


def cosine_distance(y_true,y_pred) :
	def _squared_magnitude(x):
		return T.sum( T.sqr(x) )

	return  1 - ( T.sum(y_true*y_pred)/( T.sqrt( _squared_magnitude(y_true) )*T.sqrt( _squared_magnitude(y_pred) ) )  )[()]


sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd)

model.fit(NN_in, sol_class, nb_epoch=20,verbose = 2)
#score = model.evaluate(X_test, y_test, batch_size=16)
guess = model.predict(NN_in)
cost_arr = -numpy.log( guess )
guess_choice = numpy.argmin( cost_arr ,axis = 1)
truth_arr = numpy.argmax( sol_class ,axis = 1)
correct_total = guess_choice == truth_arr
accuracy = float( sum(correct_total) )/len(guess)
print "training accuracy = " + str( accuracy )

#=================test===================================
test_guess = model.predict(NN_test)
test_entropy = -numpy.log( test_guess )
test_choice = numpy.argmin( test_entropy ,axis = 1)
test_id = X_test_dict.keys()

with open('train.csv','wb') as f :
	csv_writer = csv.writer( f )
	csv_writer.writerow( ['q_id','ans'] )
	for idx,A_E_index in enumerate(guess_choice) :
		csv_writer.writerow( A_to_E[ A_E_index ]  )


with open('final.csv','wb') as f :
	csv_writer = csv.writer( f )
	csv_writer.writerow( ['q_id','ans'] )
	for idx,A_E_index in enumerate(test_choice) :
		csv_writer.writerow( [ test_id[ idx ][1],A_to_E[ A_E_index ] ] )


