import config
import classifier
import models 
import pandas as pd
from objdict import ObjDict

def experiment(name, 
			   classifier, 
			   model_creator, 
			   model_arg_generator, 
			   output = 'experiment/{}_result.csv'):
    result = pd.DataFrame(columns=['conv_output_dimension', 
                                    'kernel_size',
                                    'pool_size',
                                    'dense_layer_count',
                                    'dense_layer_size',
                                    'dropout',
                                    'optimizer',
                                    'layer_activation',
                                    'output_activation',
                                    'score'])
    
    for  args in model_arg_generator:
        _, score   = classifier.build_model("experiment", 
		                                    model_creator = model_creator, 
						   				    **args)
        args.score = score
        result     = result.append(args, ignore_index=True)
        # write the result to file if output is supplied
        if output:
            result.to_csv(output.format(name), 
			              encoding = 'utf-8', 
						  index    = False)
    return result

def model3_arg_gen(conv_output_dimension = [None], 
              kernel_size       = [5], 
              pool_size         = [5],
              dense_layer_count = [3],
              dense_layer_size  = [512],
              dropout           = [0.3],
              optimizer         = ['adam'],
              layer_activation  = ['relu'],
              output_activation = ['sigmoid']):
    for cod in conv_output_dimension:
        for ks in kernel_size:
            for ps in pool_size:
                for dlc in dense_layer_count:
                    for dls in dense_layer_size:
                        for dp in dropout:
                            for op in optimizer:
                                for la in layer_activation:
                                    for oa in output_activation:
                                        args = ObjDict(conv_output_dimension = cod,
                                                       kernel_size           = ks,
                                                       pool_size             = ps,
                                                       dense_layer_count     = dlc,
                                                       dense_layer_size      = dls,
                                                       dropout               = dp,
                                                       optimizer             = op,
                                                       layer_activation      = la,
                                                       output_activation     = oa
                                                       )
                                        yield args
