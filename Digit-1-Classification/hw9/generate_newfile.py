import random

num_lines = sum(1 for line in open('ZipDigits.combine'))
f = open('ZipDigits.combine', 'r')
train = open('new.train','w')
test = open('new.test','w')
s = set()

while len(s) < 300:
	i = random.randint(0, num_lines)
	if i not in s:
		s.add(i)

i = 0
for line in f:
	if i in s:
		train.write(line)
	else:
		test.write(line)
	i += 1
