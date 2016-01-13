import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os
import argparse
import numpy as np
from scipy import stats
import pickle

runtags=[]

recall_type=3 #unweighted recall line[3], weighted recall line[4], for now, unweighted recall first....

#get system runtag list
for filename in os.listdir('run_results/'):
   runtags.append(filename)

recall=[]
precision=[]


file_input=open('fuse_recall_results_on_official_clusters_by_cluster_size_union.txt','r')
lines=file_input.readlines()
for i in range(1,len(lines)):
   line=lines[i].strip().split()
   if line[2]=="all":
      recall.append(line[recall_type])
      precision.append(line[5])

plt.plot(recall, precision, 'ro')


x=np.arange(0.01,1.1,0.01)
y=np.arange(0.01,1.1,0.01)
X,Y=np.meshgrid(x,y)
Z=2*X*Y/(X+Y)

CS = plt.contour(X,Y,Z,np.arange(0.05,0.50,0.05),colors='b')
plt.clabel(CS, inline=1, fontsize=10)


plt.axis([0, 1.0, 0, 1.0])
plt.xticks(np.arange(0,1.1,0.2))
plt.yticks(np.arange(0,1.1,0.2))

plt.savefig('Precision vs. Unweighted Recall_intersection.png')
plt.show()
plt.close()




# difficulty='simple'
# shutup='no'
# metrics='ELG'
# if shutup=="no":
#    recall_path=metrics+'_results_without_shutdown'
# elif shutup=="yes":
#    recall_path=metrics+'_results_with_shutdown'
# # normalized=args.normalized
# team_draft='simple_interleave'
# team_draft2='simple_interleave'

# dot={}
# this_color={}
# this_marker={}
# dot['all_pairs']={}
# this_color['all_pairs']='b'
# this_marker['all_pairs']='s'
# dot['inter_systems']={}
# this_color['inter_systems']='g'
# this_marker['inter_systems']='^'
# dot['intra_systems']={}
# this_color['intra_systems']='r'
# this_marker['intra_systems']='o'
# # credit=args.credit
# cluster="official"
# color="no"
# recall_cluster="official"
# if recall_cluster=="alternative":
#    on_alternative_clusters="_vs_recall_on_"+recall_cluster
# elif recall_cluster=="official":
#    on_alternative_clusters=""



# file_qrels_path="qrels.txt"
# qrels_dt = {}
# file_qrels = open(file_qrels_path, "r")
# lines = file_qrels.readlines()
# for line in lines:
#     line = line.strip().split()
#     topic_ind = str("MB"+line[0])

#     if topic_ind not in qrels_dt:
#       qrels_dt[topic_ind] = 1
# file_qrels.close()
# topics=sorted(qrels_dt.keys())
# # print topics


# # if normalized=="yes":
# #    file_input=open('interleaving_results/result_'+team_draft+'_'+difficulty+'_vs_'+recall_path+'_normalized.txt','r')
# # elif normalized=="no":
# #    
# run_time=['1','2','3']
# for rate in not_throw_away_rate:
#    for all_pairs in ['all_pairs','inter_systems','intra_systems']:
#       dot[all_pairs][rate]=0.0
#       for time in run_time:

#          count_DY={}
#          count_DN={}
#          count_TY={}
#          count_TN={}

#          count_DY_total=0
#          count_DN_total=0
#          count_TY_total=0
#          count_TN_total=0

#          for i in topics:
#             # rgb[i]=[np.random.rand(),np.random.rand(),np.random.rand()]
#             count_DY[i]=0
#             count_DN[i]=0
#             count_TY[i]=0
#             count_TN[i]=0

#          file_input=open('interleaving_results_'+time+'/result_'+team_draft+'_'+difficulty+'_vs_'+recall_path+'_raw_'+str(rate)+"_1.0"+'.txt','r')
#          print all_pairs,'interleaving_results_'+time+'/result_'+team_draft+'_'+difficulty+'_vs_'+recall_path+'_raw_'+str(rate)+"_1.0"+'.txt'
#          # print('interleaving_results/result_'+difficulty+'_vs_'+recall_path+'.txt')
#          lines=file_input.readlines()
#          recalls=[]
#          gains=[]
#          topic_dot=[]
#          percentage=[]
#          for i in range(1,len(lines)):
#             line=lines[i].strip().split()
#             flag=True

#             countA=long(line[3])
#             countB=long(line[4])
#             count=long(line[5])

#             if all_pairs=="all_pairs":
#                flag=True

#             elif all_pairs=="inter_systems":
#                for group_cluster in group_clusters:
#                   # print line[0],line[1],group_cluster
#                   if line[0] in group_cluster and line[1] in group_cluster:
#                      flag=False
#             elif all_pairs=="intra_systems":
#                for group_cluster in group_clusters:
#                   if line[0] in group_cluster and (line[1] not in group_cluster):
#                      flag=False

#             if team_draft2=="team_draft":
#                # if (min(countA,countB)!=0) and (float(countA-countB)/min(countA,countB)<=0.2) and (countA>=40) and (countB>=40):
#                if (min(countA,countB)==0) or (float(countA-countB)/min(countA,countB)>0.2) or (countA<40) or (countB<40):
#                   flag=False

#             if flag==True:
#                recalls.append(float(line[9]))
#                gains.append(float(line[10]))
#                if int(line[5])!=0:
#                   percentage.append(1.0*int(line[5])/(int(line[3])+int(line[4]))*100)
#                topic_num=line[2]
#                topic_dot.append(topic_num)

#                if line[14]=='DY':
#                   count_DY[topic_num]=count_DY[topic_num]+1
#                   count_DY_total=count_DY_total+1
#                elif line[14]=='DN':
#                   count_DN[topic_num]=count_DN[topic_num]+1
#                   count_DN_total=count_DN_total+1
#                elif line[14] == 'TY':
#                   count_TY[topic_num]=count_TY[topic_num]+1
#                   count_TY_total=count_TY_total+1
#                elif line[14] == 'TN':
#                   count_TN[topic_num]= count_TN[topic_num]+1
#                   count_TN_total=count_TN_total+1

#          count_total=count_DY_total+count_DN_total+count_TY_total+count_TN_total

#          # print len(recalls)
#          # print count_total

#          # print "AD "+str(float(count_DY_total)/count_total*100)+"%"
#          # print "AN "+str(float(count_TY_total)/count_total*100)+"%"
#          # print "AT "+str(float(count_DY_total+count_TY_total)/count_total*100)+"%"

#          dot[all_pairs][rate]+=float(count_DY_total+count_TY_total)/count_total*100

#          # print "DD "+str(float(count_DN_total)/count_total*100)+"%"
#          # print "DN "+str(float(count_TN_total)/count_total*100)+"%"
#          # print "DT "+str(float(count_DN_total+count_TN_total)/count_total*100)+"%"

#          # not_agree=[]
#          # for i in topics:
#          #    topic_total=count_DY[i]+count_DN[i]+count_TY[i]+count_TN[i]
#          #    # print i, topic_total e.g. MB242 666 (=37*36/2)
#          #    if topic_total!=0:
#          #       topic_agree=float(count_DY[i]+count_TY[i])/topic_total
#          #       topic_not_agree=float(count_DN[i]+count_TN[i])/topic_total      
#          #       # file_result.write(i+"\t"+str(count_DY[i])+"\t"+str(count_DN[i])+"\t"+str(count_TY[i])+"\t"+str(count_TN[i])+"\t"+str(topic_total)+"\t{0:.3f}%".format(topic_agree*100)+"\t{0:.3f}%".format(topic_not_agree*100)+"\n")
#          #       not_agree.append(topic_not_agree)

#          # mean, sigma = np.mean(not_agree), np.std(not_agree)
#          # conf_int = stats.norm.interval(0.95, loc=mean, scale=sigma/np.sqrt(len(not_agree))) #95% confidence interval
#          # # print conf_int
#          # percentage_mean,percentage_sigma=np.mean(percentage),np.std(percentage)
#          # percentage_conf_int = stats.norm.interval(0.95,loc=percentage_mean,scale=percentage_sigma/np.sqrt(len(percentage)))
#          # # print len(percentage),percentage_mean
#          # print "Percentage mean: {0:.1f}%".format(percentage_mean)+" ({0:.1f}%,".format(percentage_conf_int[0])+"{0:.1f}%)".format(percentage_conf_int[1])

#          # num_bins=10

#          # print "Agree: {0:.1f}%".format(float(count_DY_total+count_TY_total)/count_total*100)+" ({0:.1f}%,".format(100-conf_int[1]*100)+"{0:.1f}%)".format(100-conf_int[0]*100)
#          # print "Not agree: {0:.1f}%".format(float(count_DN_total+count_TN_total)/count_total*100)+" ({0:.1f}%,".format(conf_int[0]*100)+"{0:.1f}%)".format(conf_int[1]*100)

#          file_input.close()
#       dot[all_pairs][rate]=dot[all_pairs][rate]/3

# fig, ax = plt.subplots()
# for all_pairs in ['all_pairs','inter_systems','intra_systems']:
#    ax.plot(sorted([x*100 for x in not_throw_away_rate]),sorted(dot[all_pairs].values()),marker=this_marker[all_pairs],ls='-',mec=this_color[all_pairs],label=all_pairs,markersize=6,linewidth=1.5)
#    # print rate, dot[rate]['all_pairs']
# plt.xticks(np.arange(0,100+1,10))
# plt.yticks(np.arange(0,100+1,10))
# plt.grid()

# plt.xlabel("Retention probability p",fontsize=15)
# plt.ylabel("Simulation accuracy",fontsize=15)
# plt.title("Accuracy vs effort for "+metrics,fontsize=15)
# # Now add the legend with some customizations.
# ax.legend(loc=4,numpoints=1)

# # for label in legend.get_texts():
# #     label.set_fontsize(15)

# plt.savefig('plot/Prospective_'+metrics+'_on_retention_probability.png')
# plt.savefig('plot/Prospective_'+metrics+'_on_retention_probability.pdf')
# plt.show()
# plt.close()