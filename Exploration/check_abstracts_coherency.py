from tqdm import tqdm

def get_id(file_str):
    return int(file_str.split("----")[0])


with open("data/abstracts.txt", encoding="utf8") as file:
    texts = file.readlines()
text_ids= [get_id(t) for t in texts]
with open("data/author_papers.txt", encoding="utf8") as file:
    auth_papers = file.read()
    
not_found=0
for i in range(len(text_ids)):
   if  ((auth_papers.find("-"+str(text_ids[i])+"-")==-1) and (auth_papers.find(":"+str(text_ids[i])+"-")==-1) and (auth_papers.find("-"+str(text_ids[i])+"\n")==-1)):
      not_found+=1
      if i%10==0:
          print(not_found*100/i)
not_found/=len(text_ids)
print(not_found*100)