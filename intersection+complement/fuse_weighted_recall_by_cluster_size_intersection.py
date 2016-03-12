#This file is to take run file (as an input argument) and ground truth non-redundant tweets 
#to compute the unweighted precision, recall and weighted precision per topic.
import json
from sets import Set
import argparse

parser = argparse.ArgumentParser(description='Tweet Timeline Generation (TTG) evaluation script (version 1.0)')
parser.add_argument('-q', required=True, metavar='qrels', help='qrels file')
parser.add_argument('-c', required=True, metavar='clusters', help='cluster anotations')
# parser.add_argument('-r', required=True, metavar='run', help='run file')
parser.add_argument('-a', required=True, metavar='runtag1',help='run tag')
parser.add_argument('-b', required=True, metavar='runtag2',help='run tag')

args = parser.parse_args()
file_qrels_path = vars(args)['q']
clusters_path = vars(args)['c']
runtag1=vars(args)['a']
runtag2=vars(args)['b']

run_path1 = "../run_results/"+runtag1
run_path2 = "../run_results/"+runtag2
recall_path="fuse_recall_results_on_official_clusters_by_cluster_size_intersection.txt"
#Take qrels to generate dictionary of {topic number:{tweetid:weight}} 
#where weight is 0(non-relevant), 1(relevant), 2(highly relevant)
qrels_dt = {}
file_qrels = open(file_qrels_path, "r")
lines = file_qrels.readlines()
for line in lines:
   line = line.strip().split()
   topic_ind = line[0]
   if topic_ind not in qrels_dt:
      qrels_dt[topic_ind] = {}
   qrels_dt[topic_ind][line[2]] = line[3]

#Take run file and generate dictionary of {topic number:Set of tweetids for that topic}
clusters_run_dt1 = {}
clusters_run_dt2 = {}
file_run1 = open(run_path1, "r")
file_run2 = open(run_path2, "r")
lines = file_run1.readlines()
for line in lines:
   line = line.strip().split()
   topic_ind = line[0][line[0].index("MB") + 2:]
   if topic_ind not in clusters_run_dt1:
      clusters_run_dt1[topic_ind] = Set()
   clusters_run_dt1[topic_ind].add(line[2])
lines = file_run2.readlines()
for line in lines:
   line = line.strip().split()
   topic_ind = line[0][line[0].index("MB") + 2:]
   if topic_ind not in clusters_run_dt2:
      clusters_run_dt2[topic_ind] = Set()
   clusters_run_dt2[topic_ind].add(line[2])

clusters_run_dt={}
for i in clusters_run_dt1:
   if i in clusters_run_dt2:
      clusters_run_dt[i]=clusters_run_dt1[i] & clusters_run_dt2[i]
   else:
      # clusters_run_dt[i]=clusters_run_dt1[i]
      clusters_run_dt[i]=Set()
for i in clusters_run_dt2:
   if i not in clusters_run_dt1:
      # clusters_run_dt[i]=clusters_run_dt2[i]
      clusters_run_dt[i]=Set()
#Take ground truth, generate dictionary of {topic number:2D array of clusters of tweetids}, for each topic,
#compare tweet from each cluster with that from run file and compute unweighted precision, recall and weighted recall.
clusters_dt = {}
precision_total = 0
unweighted_recall_total = 0 
weighted_recall_total = 0
file_clusters = open(clusters_path, "r")
data = json.load(file_clusters)
topics = data["topics"]

file_write=open(recall_path,'a')
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
   for cluster in clusters_dt[topic_ind]:
      weight = 0
      hit_flag = 0
      for tweet in cluster:
         weight = weight + int(qrels_dt[topic_ind][tweet])
         if tweet in clusters_run_dt[topic_ind]:
            hit_flag = 1
      total_weight = total_weight + weight
      if hit_flag == 1:
         credits = credits + weight
         hit_num = hit_num + 1
         hit_flag = 0
   if len(clusters_run_dt[topic_ind])>0:
      precision = float(hit_num) / len(clusters_run_dt[topic_ind])
   else:
      precision=0
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
file_clusters.close()
file_write.close()
