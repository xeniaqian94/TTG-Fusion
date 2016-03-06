import json
from sets import Set
import argparse
import os
import timeit
import numpy as np
import codecs

#usage: python fusion_union.py -q qrels2014.txt -c cluster_official.json -o fuse_union.txt -r run_results_sub -t Data/raw_tweet_with_cleaned_tokens


# Added tokenizer tag, for use ACL or TREC tokenizer
#Return tweet dict
def extractTweet(tweetFile, tokenizer='cleaned_tokens'):
    tweetDict={}
    with open(tweetFile) as tweet_file:    
        data = json.load(tweet_file)

    for tweetId in data:
        if tokenizer == 'TREC':
            tokenList = data[tweetId]["trec_tokens"]
        else: # default
            tokenList = data[tweetId]["cleaned_tokens"]
        
        tweetDict[tweetId]=tokenList
    
    return tweetDict

def groundtruthDeduplicate(list_before_filter,clusters_dt,topic_ind):
   belong_cluster={}

   for tweetid in list_before_filter:
      belong_cluster[tweetid]=-1
   cluster_index=0
   for cluster_index in range(0,len(clusters_dt[topic_ind])):
      for tweet in clusters_dt[topic_ind][cluster_index]:
         if tweet in list_before_filter:
            belong_cluster[tweet]=cluster_index

   fused_run_dt=[]
   for tweet in list_before_filter:
      flag=1
      for previous_tweet in fused_run_dt:
         # if topic_ind=="171":
         #    print belong_cluster[tweet],belong_cluster[previous_tweet],(belong_cluster[tweet]!=-1 and belong_cluster[tweet]==belong_cluster[previous_tweet])
         if (belong_cluster[tweet]!=-1 and belong_cluster[tweet]==belong_cluster[previous_tweet]):
            flag=0
      if flag==1:
         fused_run_dt.append(tweet)
   return fused_run_dt


def calRecallPrecision(runtag1,runtag2,topics,file_write,tweetDict):
   run_path1 = "run_results/"+runtag1
   run_path2 = "run_results/"+runtag2
   clusters_run_dt = {}
   file_run1 = open(run_path1, "r")
   file_run2 = open(run_path2, "r")
   lines = file_run1.readlines()
   for line in lines:
      line = line.strip().split()
      topic_ind = line[0][line[0].index("MB") + 2:]
      if topic_ind not in clusters_run_dt:
         clusters_run_dt[topic_ind] = Set()
      clusters_run_dt[topic_ind].add(line[2])
      
   lines = file_run2.readlines()
   for line in lines:
      line = line.strip().split()
      topic_ind = line[0][line[0].index("MB") + 2:]
      if topic_ind not in clusters_run_dt:
         clusters_run_dt[topic_ind] = Set()
      if line[2] not in clusters_run_dt[topic_ind]:
         clusters_run_dt[topic_ind].add(line[2])

   precision_total = 0
   unweighted_recall_total = 0 
   weighted_recall_total = 0
   for topic in sorted(topics.keys()):
      total_weight = 0
      credits = 0
      hit_num = 0
      topic_ind = topic[line[0].index("MB") + 2:]
      topic_ind = topic_ind.encode("utf-8")
      clusters_json = topics[topic]["clusters"]
      for i in range(len(clusters_json)):
         clusters_json[i] = [s.encode("utf-8") for s in clusters_json[i]]
      clusters_dt[topic_ind] = clusters_json


      fused_run_dt=groundtruthDeduplicate(clusters_run_dt[topic_ind],clusters_dt,topic_ind)
      # fused_run_dt=clusters_run_dt[topic_ind]

      # fused_run_dt=gloveDeduplicate(clusters_run_dt[topic_ind],tweetDict)


      for cluster in clusters_dt[topic_ind]:
         weight = 0
         hit_flag = 0
         for tweet in cluster:
            weight = weight + int(qrels_dt[topic_ind][tweet])
            # if tweet in clusters_run_dt[topic_ind]:
            if tweet in fused_run_dt:
               hit_flag = 1
         total_weight = total_weight + weight
         if hit_flag == 1:
            credits = credits + weight
            hit_num = hit_num + 1
            hit_flag = 0
      # precision = float(hit_num) / len(clusters_run_dt[topic_ind])
      precision = float(hit_num) / len(fused_run_dt)
      unweighted_recall = float(hit_num) / len(clusters_dt[topic_ind])
      weighted_recall = float(credits) / total_weight
      precision_total = precision_total + precision
      unweighted_recall_total = unweighted_recall_total + unweighted_recall
      weighted_recall_total = weighted_recall_total + weighted_recall
      # file_write.write(run_path1[run_path1.rindex("/") + 1:].ljust(16) + "\t" + run_path2[run_path2.rindex("/") + 1:].ljust(16) + "\tMB" + str(topic_ind) + "\t" + "%12.4f" % unweighted_recall + "\t" + "%12.4f" % weighted_recall + "\t" + "%10.4f" % precision+"\n")
   precision_mean = precision_total / len(clusters_dt)
   unweighted_recall_mean = unweighted_recall_total / len(clusters_dt)
   weighted_recall_mean = weighted_recall_total / len(clusters_dt)
   file_write.write(run_path1[run_path1.rindex("/") + 1:].ljust(16) + "\t" + run_path2[run_path2.rindex("/") + 1:].ljust(16) + "\tall".ljust(5) + "\t" + "%12.4f" % unweighted_recall_mean + "\t" + "%12.4f" % weighted_recall_mean + "\t" + "%10.4f" % precision_mean+"\n")
   file_run1.close()
   file_run2.close()

def load_bin_vec(fname):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')

            if line%10000==0:
               print line
            # if word in vocab:
            #    word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            # else:
            #     f.read(binary_len)
    return word_vecs

def build_word_vector_matrix(vector_file, n_words):
        '''Read a GloVe array from sys.argv[1] and return its vectors and labels as arrays'''
        # numpy_arrays = []
        # labels_array = []
        word_vecs = {}
        with codecs.open(vector_file, 'r', 'utf-8') as f:
                for c, r in enumerate(f):
                        sr = r.split()
                        if len(sr)>0:
                           word_vecs[sr[0]]=np.array([float(i) for i in sr[1:]]) 
                           # labels_array.append(sr[0])
                           # numpy_arrays.append( np.array([float(i) for i in sr[1:]]) )
                        if c%10000==0:
                           print c/10000,sr[0]
                        if c == n_words:
                                return word_vecs

        # return numpy.array( numpy_arrays ), labels_array
        return word_vecs

if __name__=="__main__":
   print "inside main"
   parser = argparse.ArgumentParser(description='Tweet Timeline Generation (TTG) evaluation script (version 1.0)')
   parser.add_argument('-q', required=True, metavar='qrels', help='qrels file')
   parser.add_argument('-c', required=True, metavar='clusters', help='cluster anotations')
   # parser.add_argument('-r', required=True, metavar='run', help='run file')
   parser.add_argument('-o', required=True, metavar='recall_path',help='run tag')
   parser.add_argument('-r',required=True,metavar='runresults',help='runresults')
   parser.add_argument('-t', required=True, metavar='tweetfile', help='tweetfile')

   args = parser.parse_args()
   file_qrels_path = vars(args)['q']
   clusters_path = vars(args)['c']
   recall_path=vars(args)['o']
   run_results=vars(args)['r']
   tweetFile=vars(args)['t']

   qrels_dt = {}
   file_qrels = open(file_qrels_path, "r")
   lines = file_qrels.readlines()
   for line in lines:
      line = line.strip().split()
      topic_ind = line[0]
      if topic_ind not in qrels_dt:
         qrels_dt[topic_ind] = {}
      qrels_dt[topic_ind][line[2]] = line[3]

   #Take ground truth, generate dictionary of {topic number:2D array of clusters of tweetids}, for each topic,
   #compare tweet from each cluster with that from run file and compute unweighted precision, recall and weighted recall.
   clusters_dt = {}
   file_clusters = open(clusters_path, "r")
   data = json.load(file_clusters)
   topics = data["topics"]

   tweetDict = extractTweet(tweetFile)
   print len(tweetDict)
   file_write=open(recall_path,'w')
   file_write.write("runtag1".ljust(16) + "\t"+"runtag2".ljust(16) + "\ttopic\tunweighted_recall weighted_recall precision\n")

   runtags=[]

   #get system runtag list
   for filename in os.listdir(run_results):
      runtags.append(filename)
   # recall_path="fuse_recall_results_on_official_clusters_by_cluster_size_union.txt"
   runtags.remove('.DS_Store')
   start = timeit.default_timer()
   for i in range(len(runtags)):
      for j in range(i+1,len(runtags)):
         print runtags[i],runtags[j]
         calRecallPrecision(runtags[i],runtags[j],topics,file_write,tweetDict)

   stop = timeit.default_timer()
   print str(stop - start)+" seconds"

   file_clusters.close()
   file_write.close()

   # w2v=load_bin_vec('glove.twitter.27B.25d.txt') 

   word_vecs=build_word_vector_matrix('glove.twitter.27B.25d.txt',1193515)


   print "word2vec loaded!"
   print "num words already in word2vec: " + str(len(word_vecs))

   print word_vecs['<user>']
   print " "
   




   