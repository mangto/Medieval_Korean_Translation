from copy import copy
import re

with open(".\\gasa.txt", "r", encoding='utf8') as file:
    data = file.read()

DataLines = data.splitlines()
out = "orginal,translation"

numbers = '1234567890'

def CleanString(string:str) -> str:
    if(")" in string):
        idx = string.find(")")
        IsNum = True
        starts = copy(idx)

        while IsNum:
            starts -= 1
            if (string[starts] not in numbers): IsNum = False
        
        string = string[:starts+1] + string[string.find(")")+1:]

        return CleanString(string)

    else:
        string = re.sub(',', '',string)
        string = re.sub(r'[^가-힣0-9 ]', '', string)
        return string

for idx in range(len(DataLines)//2):
    out += f"\n{DataLines[2*idx]},{CleanString(DataLines[2*idx+1])}"

with open(".\\translation.csv", "w", encoding='utf8') as file:
    file.write(out)