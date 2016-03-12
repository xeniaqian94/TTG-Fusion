import os

runtags=[]

#get system runtag list
for filename in os.listdir('../run_results/'):
   runtags.append(filename)
recall_path="fuse_weighted_recall_by_cluster_size_intersection.txt"
file_write=open(recall_path,'w')
file_write.write("runtag1".ljust(16) + "\t"+"runtag2".ljust(16) + "\ttopic\tunweighted_recall weighted_recall precision\n")
file_write.close()
runtags.remove('.DS_Store')
for i in range(len(runtags)):
   for j in range(i+1,len(runtags)):
      print runtags[i],runtags[j]

      os.system(str("python fuse_weighted_recall_by_cluster_size_intersection.py -q ../qrels2014.txt -c ../cluster_official.json -a "+runtags[i]+" -b "+runtags[j]+" -r "+recall_path))