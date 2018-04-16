import numpy as np
import tensorflow as tf
import os
from genomelake.extractors import ArrayExtractor

DEFER_DELETE_SIZE=int(250 * 1e6)
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
session_config = tf.ConfigProto()
session_config.gpu_options.deferred_deletion_bytes = DEFER_DELETE_SIZE
session_config.gpu_options.per_process_gpu_memory_fraction = 0.45
sess = tf.Session(config=session_config)

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
    #Calculates the locations that activate the relu after colvolution and bias
    
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
        if counts>0:
            pfm/=(counts+0.0)
            info_list.append(get_information_content_matrix(pfm_with_pseudocount(pfm)))
    return info_list

def pfm_counts_pipeline(intervals_bedtool,genome_path,filters,biases,num_intervals):
    genome_extractor = ArrayExtractor(genome_path)
    print("Creating intervals list \n")
    intervals_list = [intervals_bedtool[i] for i in range(num_intervals)]
    extracted_intervals = genome_extractor(intervals_list)
    assert extracted_intervals.shape[0] == num_intervals
    assert extracted_intervals.shape[2] == filters.shape[1]   #Number of bases must match
    print("Reshaping Intervals and filters to use tf.nn.conv2D")
    extracted_intervals_reshape = np.expand_dims(extracted_intervals.transpose((0,2,1)),axis=-1)
    filters_reshape = np.expand_dims(filters,axis=-1).transpose((1,0,3,2))
    pfm_counts = get_counts_list(intervals=extracted_intervals_reshape,filters=filters_reshape,biases=biases)
    info_matrices = information_matrix_pwms(pfm_counts)
    return pfm_counts,info_matrices
   