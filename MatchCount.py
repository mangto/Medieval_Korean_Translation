with open(".\\translation.csv", "r", encoding='utf8') as file:
    data = file.read().splitlines()[1:]

valid = 0
for line in data:
    orginal, translation = line.split(',')
    valid += len(orginal.split(' ')) == len(translation.split(' '))

print(f'Valid: {valid}/{len(data)}')