from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential

# DEPRECATED 
def model1(trainer, 
		   output_activation     = 'sigmoid'):
    print("Building CNN model...")
    model       = Sequential()
    num_classes = len(trainer.classes)
    model.add(Embedding(trainer.get_vocabulary_size(), 
                        trainer.config.runtime.embedding_dim, 
                        weights      = [trainer.embedding_matrix], 
                        input_length = trainer.max_input_len, 
                        trainable    = False)
              )
	# Sequence classification with 1D convolutions for text input:
	# REMEMBER TO CONTINUE EDIT THIS
	# CHANGE THE NUMBER OF INPUT https://richliao.github.io/supervised/classification/2016/11/26/textclassifier-convolutional/
	# ADD MORE COMMENT, XEM FILE smallervgget.py
	# UNDERSTAND CONVOLUTION IN TEXT: http://debajyotidatta.github.io/nlp/deep/learning/word-embeddings/2016/11/27/Understanding-Convolutions-In-Text/
    
	# CONV => RELU => POOL
    model.add(Conv1D( 128, 5, activation = 'relu'))
    
	# add BatchNormalization
	# model.add(BatchNormalization(axis = 1))
    model.add(MaxPooling1D(5)) # or: model.add(MaxPooling1D(5,1)) 
	
	# FLATTEN
    model.add(Flatten())
	
	# first HIDDEN LAYER => RELU layers
    model.add(Dense(128, activation = 'relu'))
   
	
	# second HIDDEN LAYER => RELU layers
    model.add(Dense(128, activation = 'relu'))
   
	
	# OUTPUT LAYER
    model.add(Dense(num_classes, activation = output_activation))
    		
	# COMPILE CNN MODEL
    model.compile(optimizer     = 'adam', 
                    loss        = 'binary_crossentropy', 
                    metrics     = ['acc'])
    print(model.summary())
    return model

# DEPRECATED 
def model2(trainer, output_activation     = 'sigmoid'):
    print("Building CNN model...")
    model       = Sequential()
    num_classes = len(trainer.classes)
    model.add(Embedding(trainer.get_vocabulary_size(), 
                        trainer.config.runtime.embedding_dim, 
                        weights      = [trainer.embedding_matrix], 
                        input_length = trainer.max_input_len, 
                        trainable    = False)
              )
			  
	# CONV => RELU => POOL
    model.add(Conv1D(128, 5, activation = 'relu') ) 
    model.add(MaxPooling1D(5, 1))
	
	# FLATTEN
    model.add(Flatten())
	
	# first HIDDEN LAYER => RELU layers
    model.add(Dense(512, activation = 'relu') )
    model.add(Dropout(0.2))
	
	# second HIDDEN LAYER => RELU layers
    model.add(Dense(512, activation = 'relu') )
    model.add(Dropout(0.3))
	
	# second THIRD LAYER => RELU layers
    model.add(Dense(512, activation = 'relu') )
    model.add(Dropout(0.3))
	
	# OUTPUT LAYER
    model.add(Dense(num_classes, activation = output_activation))
	
	# COMPILE CNN MODEL
    model.compile( optimizer   = 'adam', 
                   loss        = 'binary_crossentropy', 
                   metrics     = ['acc']
				  )
    print(model.summary())
    return model

# IN USED  
def model3(trainer, 
           conv_output_dimension  = None, 
           kernel_size           = 5, 
           pool_size             = 5,
           dense_layer_count     = 3,
           dense_layer_size      = 512,
           dropout               = 0.3,
           optimizer             = 'adam',
           layer_activation      = 'relu',
           output_activation     = 'sigmoid'):
    print("Building CNN model...")
    model                 = Sequential()
    num_classes           = len(trainer.classes)
    
    model.add(Embedding(trainer.get_vocabulary_size(), 
                        trainer.config.runtime.embedding_dim, 
                        weights      = [trainer.embedding_matrix], 
                        input_length = trainer.max_input_len, 
                        trainable    = False)
                )
				
	# CONV => RELU => POOL		
	# model.add(Conv1D(128, 5, activation = 'relu') ) 	
    model.add(Conv1D(128, kernel_size, activation = layer_activation))
    model.add(MaxPooling1D(pool_size, 1))
	# MaxPooling1D(pool_size=2, strides=None, padding='valid')
	# strides: Integer, or None. Factor by which to downscale. If None, it will default to pool_size.
	
	
	# FLATTEN
    model.add(Flatten())
    
	# first and n HIDDEN LAYERS => RELU layers
    n = dense_layer_count
    while n > 0:
        model.add(Dense(dense_layer_size, activation = layer_activation))
        if dropout > 0:
            model.add(Dropout(dropout))
        n = n - 1
		
	# OUTPUT LAYER	
    model.add(Dense(num_classes, activation = output_activation))
	
	# COMPILE CNN MODEL
    model.compile( optimizer   = optimizer, 
	               loss        = 'binary_crossentropy',  
				   metrics     = ['acc']
				  )
    print(model.summary())
    return model
  
  
  
  
  
  
  
  
  