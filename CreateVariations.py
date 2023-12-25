from random import shuffle

def Indexize(string:str, sep:str=' ') -> list[tuple]:
    OrgLen = len(string)
    IsSep = sep in string and OrgLen > 0
    idxs = []
    LastIdx = 0

    while IsSep:
        SepIdx = string.find(sep) # seperation index
        idxs.append((LastIdx, LastIdx+SepIdx))
        LastIdx = LastIdx + SepIdx + 1
        string = string[SepIdx+1:]
        IsSep = sep in string and len(string) > 0 # update

    idxs.append((LastIdx, OrgLen))

    return idxs

def IsValid(A:str, B:str, sep:str=' ') -> bool:
    # sep 기준으로 나누었을 때 어절 개수가 같은지 확인
    return len(A.split(sep)) == len(B.split(sep))

def load(string:str, token:list[tuple[int, int]], idxs:list[int], sep:str=' ') -> str:
    new = ""
    
    for idx in idxs:
        tk = token[idx]
        new += sep + string[tk[0]:tk[1]]
    new = new[1:]

    return new

def variation(A:str, B:str, sep:str=' ') -> str:
    idxs = list(range(len(A.split(sep)))) # indexes
    shuffle(idxs) # mix

    AToken = Indexize(A)
    BToken = Indexize(B)

    NewA = load(A, AToken, idxs)
    NewB = load(B, BToken, idxs)
    return NewA + "," + NewB


with open(".\\translation.csv", "r", encoding='utf-8') as file:
    data = file.read().splitlines()[1:]

VARIATIONS = 24

csv = "orginal,translation"


for line in data:
    orginal, translation = line.split(',')

    if (not IsValid(orginal, translation)): continue
    for _ in range(VARIATIONS):
        csv += "\n" + variation(orginal, translation)

with open(".\\variation.csv", "w", encoding='utf8') as file:
    file.write(csv)
