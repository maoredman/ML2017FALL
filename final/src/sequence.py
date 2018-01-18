import json
import pickle
import jieba
import sys

def sentence2words(sentence):
	#print(sentence)
	seg_list = jieba.cut(sentence)
	return " ".join(seg_list).split()

def sentence2wordset(sentence):
	words = sentence2words(sentence)
	wordset = set(words)
	return wordset

def findmaxmatch(sentences,questionset):
	maxset = set()
	maxmatch = -1
	maxid = -1
	sentenceid = 0
	for i in range(len(sentences)):
		wordset_s  = sentence2wordset(sentences[i])
		inter = wordset_s & questionset
		match = len(inter)
		if match > maxmatch:
			maxid = sentenceid
			maxmatch = match
			maxset = inter
		sentenceid += 1
	return maxmatch,maxset,maxid

def getid(maxmatch,maxid,maxset,sentences):
	idlist = []
	nowid = 0
	#print(sentences)
	for i in range(maxid):
		nowid += len(sentences[i])+1
	words = sentence2words(sentences[maxid])
	if '\n' == sentences[maxid][0]:
		nowid += 1
	print(nowid)
	for word in words:
		if word not in maxset:
			idlist.extend(list(range(nowid,nowid+len(word))))
		nowid += len(word)
	return idlist






def train():
	outfile = open('./sentenceout.csv','w')
	outfile.write('id,answer\n')
	testfile = open("../QA/test-v1.1.json",'r')
	test_raw_json = testfile.readline()
	test_rawdata = json.loads(test_raw_json)
	count = 1
	test_x = []#[context,question,questionid]

	test = 0
	for i in test_rawdata["data"]:
		for j in i["paragraphs"]:
			for m in j["qas"]:
				test+=1
				print('id = ' + m["id"])
				print('context:   '+j["context"])
				print('question: '+m["question"])
				context = j["context"].split('。')
				question = m["question"]
				wordset = sentence2wordset(question)

				maxmatch,maxset,maxid = findmaxmatch(context,wordset)
				print('maxid:    '+str(context[maxid]))
				idlist = getid(maxmatch,maxid,maxset,context)
				print(idlist)

				outfile.write('\n'+m["id"]+',') 
				for i in range(len(idlist)):
					sys.stdout.write(j["context"][idlist[i]])
					outfile.write(str(idlist[i]))
					if i != len(idlist)-1:
						outfile.write(' ')
					else:
						break
				sys.stdout.write('\n')
	print(test)
'''
testfile = open("../QA/test-v1.1.json",'r')
test_raw_json = testfile.readline()
test_rawdata = json.loads(test_raw_json)
s = test_rawdata["data"][0]["paragraphs"][1]["context"].split('。')
wordset = sentence2wordset(test_rawdata["data"][0]["paragraphs"][1]["qas"][0]["question"])
print(s)
print(findmaxmatch(s,wordset))
'''
train()