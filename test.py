
articles=[]

with open("data/author_papers.txt") as f:
    lines=f.readlines()    
    
for l in lines:
    for x in l.split(':')[1].split('\n')[0].split('-'):
        articles.append(x)

with open("data/abstracts.txt", encoding="utf8") as file:
    abstracts = file.readlines()

ab=[x.split("---")[0] for x in abstracts]

not_found=0
i=0

from tqdm import tqdm

for a in articles:
    i+=1
    freq=not_found/i
    try:
        _=ab.index(a)
    except ValueError:
         not_found+=1
         print(a, freq)





