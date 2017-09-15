import sys

with open(sys.argv[1]) as f:
    data = f.read()
    data = data.split() # splits by whitespace chars
    word_list = []
    word_count={}
    for word in data:
        if word not in word_count:
            word_count[word] = 1
            word_list.append(word)
        else:
            word_count[word] += 1

    word_list[-1] = word_list[-1].strip() # strips all whitespace chars; last word has trailing \n
    
    with open("Q1.txt", "w") as text_file:
        for word in word_list[0:-1]:
            print(word," ",word_list.index(word)," ", word_count[word], end = "\n", file=text_file)
        
        last_word = word_list[-1]
        print(last_word," ",word_list.index(last_word)," ", word_count[last_word], end = "", file=text_file)