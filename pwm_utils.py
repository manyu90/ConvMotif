#!/bin/env python

"""
PWM utils to create pwms from Keras model conv filters.

Keras version should be 1.2.2 for shapes to be consistent with this code. Keras 2 uses tensorflow conv1D as a backend and the shapes sare different
The virtual environment used is pwm_utils (on the lab cluster. Need to create a setup.py file from this environment)
"""

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import shutil
import numpy as np
import tensorflow as tf
import os
from genomelake.extractors import ArrayExtractor
import keras.backend as K
from plot import seqlogo_fig
import glob
from keras.models import import model_from_yaml
from pybedtools import BedTool
import json

DEFER_DELETE_SIZE=int(250 * 1e6)
def create_tensorflow_session(visiblegpus):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(visiblegpus)
    session_config = tf.ConfigProto()
    session_config.gpu_options.deferred_deletion_bytes = DEFER_DELETE_SIZE
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.45
    session = tf.Session(config=session_config)
    K.set_session(session)
    return session

sess = create_tensorflow_session(0)


###With methylation some of these things need to be redefined (maybe only normalize on the first 4 channels if m is treated as separate)

def pwm(pfm):
    pwm = np.log(pfm/.25)
    return pwm

def pfm_with_pseudocount(pfm,pseudocount=.001):
    pfm+=pseudocount
    pfm/=(1.0+4*pseudocount)
    return pfm

def get_information_content_matrix(pfm):
    return pfm*np.log2(pfm/0.25)

def activated_locations_with_filters(intervals,filters,biases):
    #Calculates the locations that activate the relu fter colvolution and bias
    
    #intervals shape = (batch_size,height,width,num_channels) ex. (None,4,101,1)   
    #filters shape = (height,width,input_channels,out_channels) ex . (4,24,1,16)
    #biases shape = (out_channels,)  ex.(16,)
    assert len(intervals.shape) == 4
    assert len(filters.shape) == 4
    assert intervals.shape[1] == filters.shape[0]   #Heights match
    assert intervals.shape[-1] == 1  #Number of input channels = 1
    assert filters.shape[2] == 1    #Number of input channels = 1
    assert len(biases.shape) == 1
    assert biases.shape[0] == filters.shape[-1] 
    print("All shapes are compatible:")
    print("Proceeding with convolutions:")
    total_bases = filters.shape[0]
    total_filters = biases.shape[0]
    total_intervals = intervals.shape[0]
    filter_width = filters.shape[1]
    interval_width = intervals.shape[2]
    
    ##Define placeholder variables
    intervals_placeholder = tf.placeholder(shape=intervals.shape,dtype=tf.float32)
    filters_placeholder = tf.placeholder(shape=filters.shape,dtype=tf.float32)
    biases_placeholder = tf.placeholder(shape=biases.shape,dtype=tf.float32)
    ##Feed Dict
    feed_dict={intervals_placeholder:intervals,filters_placeholder:filters,biases_placeholder:biases}
    ##Convolution Op
    convolution_op = tf.nn.conv2d(input=intervals_placeholder,filter=filters_placeholder,strides=[1,1,1,1],padding='VALID')
    activated_locations_op = tf.nn.relu(convolution_op+biases_placeholder)
    activated_locations = sess.run(activated_locations_op,feed_dict=feed_dict)
    return activated_locations

 
def get_motifs_from_activations(activations,intervals,filters,biases):
    ##activations shape= (10,1,78,16)
    
    #intervals shape = (batch_size,height,width,num_channels) ex. (10,4,101,1)   
    #filters shape = (height,width,input_channels,out_channels) ex . (4,24,1,16)
    #biases shape = (out_channels,)  ex.(16,)
    intervals = intervals.squeeze()
    activations = activations.squeeze()
    total_sequences_in_batch = len(activations)
    assert filters.shape[-1] == biases.shape[0]    #This is the total filters
    total_filters = biases.shape[0]
    filter_width = filters.shape[1]
    total_bases=filters.shape[0]
    pfm_arr_list=[]
    for i in range(total_filters):
        ##Find all seqeunces that activate this specific filter
        total_activated_sequences=0
        sum_activating_sequences=np.zeros((total_bases,filter_width))
        print("Doing filter %s"%(str(i)))
        for j in range(total_sequences_in_batch):
            if np.max(activations[j,:,i])>0.0:
                max_location_in_sequence = np.argmax(activations[j,:,i])
               # print("max_location in sequence%s is %s"%(j,max_location_in_sequence))
                
                max_sequence = intervals[j,:,max_location_in_sequence:max_location_in_sequence+filter_width]
                #print(max_sequence.shape)
                assert max_sequence.shape==sum_activating_sequences.shape
                sum_activating_sequences[:,:] += max_sequence
                total_activated_sequences +=1
        
        #sum_activating_sequences/=total_activated_sequences
        pfm_arr_list.append((sum_activating_sequences,total_activated_sequences))
        
    return pfm_arr_list




def get_counts_list(intervals,filters,biases):
    """intervals: np.array of shape (batch_size,height,width,num_channels) ex. (10,4,101,1)
    filters: np.array of shape (height,width,input_channels,out_channels) ex . (4,24,1,16)
    biases: np.array of shape (out_channels,)  ex.(16,)
    return: A list of count matrices, which counts the sum of the maximally activated sequences
		Divide by counts to generate a pfm (counts might be 0, in which case, the filter was never activated) 
    """
    activated_locs = activated_locations_with_filters(intervals,filters,biases)
    return get_motifs_from_activations(activated_locs,intervals,filters,biases)


def information_matrix_pwms(pfm_counts_list):
    info_list=[]
    for elem in pfm_counts_list:
        pfm,counts=elem[0],elem[1]
        start = (len(pfm)-4)/2
        end  = start+4
        pfm_seq = pfm[start:end,:]
        if counts>0:
            pfm/=(counts+0.0)
            info_pwm = get_information_content_matrix(pfm_with_pseudocount(pfm_seq))
            pfm[start:end,:] = info_pwm
            info_list.append(pfm)

    return info_list


#This pipeleine currently works just for seq only models
#This function needs an input of filters and biases. Does not work directly with the model
def pfm_counts_pipeline(intervals_bedtool,genome_path,filters,biases,num_intervals):
    genome_extractor = ArrayExtractor(genome_path)
    print("Creating intervals list \n")
    intervals_list = [intervals_bedtool[i] for i in range(num_intervals)]
    extracted_intervals = genome_extractor(intervals_list)
    assert extracted_intervals.shape[0] == num_intervals
    assert extracted_intervals.shape[2] == filters.shape[2]   #Number of bases must match
    print("Reshaping Intervals and filters to use tf.nn.conv2D")
    extracted_intervals_reshape = np.expand_dims(extracted_intervals.transpose((0,2,1)),axis=-1)
    filters_reshape = filters.transpose((2,0,1,3))
    pfm_counts = get_counts_list(intervals=extracted_intervals_reshape,filters=filters_reshape,biases=biases)
    info_matrices = information_matrix_pwms(pfm_counts)
    return pfm_counts,info_matrices


#Edit this to take in the intervals file directly
#Should take the pos intervals file
#Add the function to create the pos intervals file directly 


def pfm_counts_pipeline_seq_models(intervals_bedtool,genome_path,model,
    conv_layer_name = 'conv_layer_1',num_intervals = 500,savedir=None):
    genome_extractor = ArrayExtractor(genome_path)
    print("Creating intervals list \n")
    intervals_list = [intervals_bedtool[i] for i in range(num_intervals)]
    extracted_intervals = genome_extractor(intervals_list)
    assert extracted_intervals.shape[0] == num_intervals
    
    #Get the conv layer from the name
    try:
        conv_layer = model.get_layer(conv_layer_name).get_weights()
    except Exception as e:
        print(e)
        return
    filters,biases = conv_layer[0],conv_layer[1]
    assert extracted_intervals.shape[2] == filters.shape[2]   #Number of bases must match
    print("Reshaping Intervals and filters to use tf.nn.conv2D")
    extracted_intervals_reshape = np.expand_dims(extracted_intervals.transpose((0,2,1)),axis=-1)
    filters_reshape = filters.transpose((2,0,1,3))
    pfm_counts = get_counts_list(intervals=extracted_intervals_reshape,filters=filters_reshape,biases=biases)
    info_matrices = information_matrix_pwms(pfm_counts)
    if savedir:
        save_pfm_counts(pfm_counts,savedir)
        save_info_matrices(info_matrices,savedir) 
    return pfm_counts,info_matrices

def pfm_counts_pipeline_seq_meth_5mC_models(intervals_bedtool,genome_path,methylation_path,model,
    conv_layer_name = 'conv_layer_1',num_intervals = 500,savedir=None):
    genome_extractor = ArrayExtractor(genome_path)   #shape should be (batch_size,1000,4) for example
    meth_extractor = ArrayExtractor(methylation_path) #shape should be (batch_size,1000)  for examples
    print("Creating intervals list \n")
    intervals_list = [intervals_bedtool[i] for i in range(num_intervals)]
    extracted_intervals_seq = genome_extractor(intervals_list)
    extracted_intervals_meth = meth_extractor(intervals_list)
    assert extracted_intervals_seq.shape[0] == num_intervals
    assert extracted_intervals_meth.shape[0] == num_intervals
    #Add in the steps to extract stranded methylation from this and generate the 6 channel intervals
    c_locations = extracted_intervals_seq[:,:,1]
    g_locations = extracted_intervals_seq[:,:,2]
    c_meth = np.expand_dims(c_locations*extracted_intervals_meth,axis=-1)
    g_meth = np.expand_dims(g_locations*extracted_intervals_meth,axis=-1)
    seq_meth_input = np.concatenate([c_meth,extracted_intervals_seq,g_meth],axis=-1)    #shape should be (batch_size,1000,6). This is the same order as the input is fed to the neural net
    #Get the conv layer from the name
    try:
        conv_layer = model.get_layer(conv_layer_name).get_weights()
    except Exception as e:
        print(e)
        return
    filters,biases = conv_layer[0],conv_layer[1]
    assert seq_meth_input.shape[2] == filters.shape[2]   #Number of bases must match. Should be = 6
    print("Reshaping Intervals and filters to use tf.nn.conv2D")
    seq_meth_intervals_reshape = np.expand_dims(seq_meth_input.transpose((0,2,1)),axis=-1)
    filters_reshape = filters.transpose((2,0,1,3))
    pfm_counts = get_counts_list(intervals=seq_meth_intervals_reshape,filters=filters_reshape,biases=biases)
    info_matrices = information_matrix_pwms(pfm_counts)
    if savedir:
        save_pfm_counts(pfm_counts,savedir)
        save_info_matrices(info_matrices,savedir) 
    return pfm_counts,info_matrices

def pfm_counts_pipeline_seq_AC_meth_models(intervals_bedtool,genome_path,C_methylation_path,A_methylation_path,model,
    conv_layer_name = 'conv_layer_1',num_intervals = 500,savedir=None):
    genome_extractor = ArrayExtractor(genome_path)   #shape should be (batch_size,1000,4) for example
    C_meth_extractor = ArrayExtractor(C_methylation_path) #shape should be (batch_size,1000)  for examples
    A_meth_extractor = ArrayExtractor(A_methylation_path) #shape should be (batcg_size,1000) for example  
    print("Creating intervals list \n")
    intervals_list = [intervals_bedtool[i] for i in range(num_intervals)]
    extracted_intervals_seq = genome_extractor(intervals_list)
    extracted_intervals_Cmeth = C_meth_extractor(intervals_list)
    extracted_intervals_Ameth = A_meth_extractor(intervals_list)

    assert extracted_intervals_seq.shape[0] == num_intervals
    assert extracted_intervals_Cmeth.shape[0] == num_intervals 
    assert extracted_intervals_Ameth.shape[0] == num_intervals  
    #Add in the steps to extract stranded methylation from this and generate the 8 channel intervals
    c_locations = extracted_intervals_seq[:,:,1]
    g_locations = extracted_intervals_seq[:,:,2]
    a_locations = extracted_intervals_seq[:,:,0]
    t_locations = extracted_intervals_seq[:,:,3]

    c_meth = np.expand_dims(c_locations*extracted_intervals_Cmeth,axis=-1)
    g_meth = np.expand_dims(g_locations*extracted_intervals_Cmeth,axis=-1)
    a_meth = np.expand_dims(a_locations*extracted_intervals_Ameth,axis=-1)
    t_meth = np.expand_dims(t_locations*extracted_intervals_Ameth,axis=-1)
    seq_meth_input = np.concatenate([a_meth,c_meth,extracted_intervals_seq,g_meth,t_meth],axis=-1)    #shape should be (batch_size,1000,6). This is the same order as the input is fed to the neural net
    #Get the conv layer from the name
    try:
        conv_layer = model.get_layer(conv_layer_name).get_weights()
    except Exception as e:
        print(e)
        return
    filters,biases = conv_layer[0],conv_layer[1]
    assert seq_meth_input.shape[2] == filters.shape[2]   #Number of bases must match. Should be = 6
    print("Reshaping Intervals and filters to use tf.nn.conv2D")
    seq_meth_intervals_reshape = np.expand_dims(seq_meth_input.transpose((0,2,1)),axis=-1)
    filters_reshape = filters.transpose((2,0,1,3))
    pfm_counts = get_counts_list(intervals=seq_meth_intervals_reshape,filters=filters_reshape,biases=biases)
    info_matrices = information_matrix_pwms(pfm_counts)
    if savedir:
        save_pfm_counts(pfm_counts,savedir)
        save_info_matrices(info_matrices,savedir) 
    return pfm_counts,info_matrices

def pfm_counts_pipeline_seq_A_meth_models(intervals_bedtool,genome_path,A_methylation_path,model,
    conv_layer_name = 'conv_layer_1',num_intervals = 500,savedir=None):
    genome_extractor = ArrayExtractor(genome_path)   #shape should be (batch_size,1000,4) for example
    A_meth_extractor = ArrayExtractor(A_methylation_path) #shape should be (batcg_size,1000) for example  
    print("Creating intervals list \n")
    intervals_list = [intervals_bedtool[i] for i in range(num_intervals)]
    extracted_intervals_seq = genome_extractor(intervals_list)
    extracted_intervals_Ameth = A_meth_extractor(intervals_list)

    assert extracted_intervals_seq.shape[0] == num_intervals
    assert extracted_intervals_Cmeth.shape[0] == num_intervals 
    assert extracted_intervals_Ameth.shape[0] == num_intervals  
    #Add in the steps to extract stranded methylation from this and generate the 8 channel intervals
    a_locations = extracted_intervals_seq[:,:,0]
    t_locations = extracted_intervals_seq[:,:,3]

    a_meth = np.expand_dims(a_locations*extracted_intervals_Ameth,axis=-1)
    t_meth = np.expand_dims(t_locations*extracted_intervals_Ameth,axis=-1)
    seq_meth_input = np.concatenate([a_meth,extracted_intervals_seq,t_meth],axis=-1)    #shape should be (batch_size,1000,6). This is the same order as the input is fed to the neural net
    #Get the conv layer from the name
    try:
        conv_layer = model.get_layer(conv_layer_name).get_weights()
    except Exception as e:
        print(e)
        return
    filters,biases = conv_layer[0],conv_layer[1]
    assert seq_meth_input.shape[2] == filters.shape[2]   #Number of bases must match. Should be = 6
    print("Reshaping Intervals and filters to use tf.nn.conv2D")
    seq_meth_intervals_reshape = np.expand_dims(seq_meth_input.transpose((0,2,1)),axis=-1)
    filters_reshape = filters.transpose((2,0,1,3))
    pfm_counts = get_counts_list(intervals=seq_meth_intervals_reshape,filters=filters_reshape,biases=biases)
    info_matrices = information_matrix_pwms(pfm_counts)
    if savedir:
        save_pfm_counts(pfm_counts,savedir)
        save_info_matrices(info_matrices,savedir) 
    return pfm_counts,info_matrices



def save_pfm_counts(counts_list,filepath):
    counts = []
    for i,item in enumerate(counts_list):
        counts.append(item[1])
        np.save(filepath+'/pfm_counts_arr%s.npy'%(str(i)),item[0])
    counts_arr = np.array(counts)
    np.save(filepath+'/pfm_counts.npy',counts_arr) 

    print("Saved pfm counts to {}\n".format(filepath))
       
def save_info_matrices(info_list,filepath):
    for i,item in enumerate(info_list):
        np.save(filepath+'/info_matrix%s.npy'%(str(i)),item)

             
    print("Saved information content matrices to {} \n".format(filepath))

"""
Plotting all information matrices : Default is the 4 channel DNA
Just change the VOCAB to get it to plot the other information matrices
"""
def plot_and_save_information_content_pwms(filepath,vocab='DNA'):
    counts = 0
    for filename in glob.glob(filepath+'/info_matrix*'):
         counts+=1
    
    if counts>0:
        for i in range(counts):
            filename_ = os.path.abspath(filepath + '/info_matrix{}.npy'.format(str(i)))
            arr_ = np.load(filename_).transpose()    #The plotting code expects the transposed shape
            fig = seqlogo_fig(arr_,vocab=vocab)
            fig_save_path = os.path.abspath(filepath + '/info_pwm{}.png'.format(str(i)))
            fig.savefig(fig_save_path)
    print("Saved all images to {}".format(filepath)) 		


"""
Filters the set of intervals to create a set of positive intervals
"""
def filter_positive_test_intervals(path_to_intervals_file,path_to_labels,test_chroms_list,path_to_positive_intervals):
    intervals_dataframe = pd.read_csv(path_to_intervals_file,sep = '\t',names=['chr','start','end','labels'])
    labels = np.load(path_to_labels)
    intervals_dataframe['labels'] = label
    filtered_pos_dataframe = intervals_dataframe.loc[intervals_dataframe['chr'].isin(test_chroms_list) & intervals_dataframe['labels']==1]
    filtered_pos_dataframe.to_csv(path_to_positive_intervals,sep='\t',header=False,index=False)
    print("Wrote to {}".format(path_to_positive_intervals))


"""
This takes in the intervals file and runs the appropriate pipeline for methylation motifs on ALL the positive intervals from the test chromosomes
This needs to be augmented/ or a new function needs to be defined which only will detect motifs on highly methylated intervals and all intervals
"""
def run_pipeline_with_all_pos_intervals(path_to_modelspec,path_to_model_arch,path_to_model_weights,path_to_intervals_file,path_to_labels,test_chroms_list,savedir=None,datasetspec_file,path_to_pos_intervals=None):
    print('Loading model \n')
    with open(path_to_seq_arch,'r') as f:
        model = model_from_yaml(f)
    model.load_weights(path_to_model_weights)
    if not path_to_pos_intervals:
        os.makedirs('./tmp_intervals/')
        path_to_pos_intervals = './tmp_intervals/pos_intervals.bed'
    #Create the filtered pos intervals set
    print("Creating positive intervals set\n")
    filter_positive_test_intervals(path_to_intervals_file,path_to_labels,test_chroms_list,path_to_positive_intervals)
    pos_intervals_bedtool = BedTool(path_to_positive_intervals) 
    num_intervals = len(pos_intervals_bedtool)
    #Load the datasetspec to get the data sources
    with open(datasetspec_file,'r') as f:
        datasetspec = json.load(f)
    
    [(celltype, data_sources)] = datasetspec.items()
    with open(path_to_modelspec,'r') as f:
        model_name = json.load(f)['model_class']
    
    path_to_genome_bcolz = data_sources['genome_data_dir']
    path_to_C_methylation_bcolz = data_sources['methylation_data_dir']
    path_to_A_methylation_bcolz = data_sources['A_methylation_data_dir']
    ##Dictionary of commands to run depending on the model type
    commands_dict = {'SequenceReverseComplementClassifier':pfm_counts_pipeline_seq_models(pos_intervals_bedtool,path_to_genome_bcolz,model,num_intervals=num_intervals,savedir=savedir) ,
            'SequenceMethylationReverseComplementClassifier':pfm_counts_pipeline_seq_meth_5mC_models(pos_intervals_bedtool,path_to_genome_bcolz,path_to_C_methylation_bcolz,model,num_intervals = num_intervals,savedir = savedir),
            'SequenceA_MethylationReverseComplementClassifier':pfm_counts_pipeline_seq_A_meth_models(pos_intervals_bedtool,path_to_genome_bcolz,path_to_A_methylation_bcolz,model,num_intervals = num_intervals, savedir = savedir),  
            'SequenceACMethylationReverseComplementClassifier':pfm_counts_pipeline_seq_AC_meth_models(pos_intervals_bedtool,path_to_genome_bcolz,path_to_C_methylation_bcolz,path_to_A_methylation_bcolz,model,num_intervals = num_intervals,savedir = savedir) 
        }   
    
    ##Run the command corresponding to the model name
    assert model_name in commands_dict
    commands_dict[model_name]
    ##Remove the tmp folder if it got created
    if os.path.exists('./tmp_intervals'):
        shutil.rmtree('./tmp_intervals')


"""
Need to implement these soon 
"""

def pipeline_for_methylated_intervals():
    pass

def get_methylated_intervals():
    pass

def discover_importance_scored_motifs():
    pass

    
