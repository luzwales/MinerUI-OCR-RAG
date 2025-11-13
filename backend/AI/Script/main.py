import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
text = " Natural Language Processing (NLP) is a field of study in computer science, artificial intelligence, and linguistics concerned with the interactions between computers and human (natural) languages."
sentences = nltk.sent_tokenize(text)
print(sentences)
# 分词示例
sentence = sentences[0]
tokens = nltk.word_tokenize(sentence)
print(tokens)
# 词性标注示例
pos_tags = nltk.pos_tag(tokens)
print(pos_tags)
# 命名实体识别示例
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
text = "Barack Obama was born in Honolulu, Hawaii. John F. Kennedy was the 35th president of the United States."
chunks = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
print(chunks)