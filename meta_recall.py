import os

runtags=[]

#get system runtag list
for filename in os.listdir('run_results/'):
   runtags.append(filename)
# runtags.remove('.DS_Store')
for i in runtags:
   os.system(str("python ttg_eval_weighted_recall_by_cluster_size.py -q qrels2014.txt -c cluster_official.json -t "+i))