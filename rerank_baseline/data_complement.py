import sys

for line in sys.stdin:
    parts = line[:-1].split(' ')
    vec_m = {}
    for o in parts[2:]:
        k, v = o.split(":")
        vec_m[int(k)] = v
    vec = []
    vec.append(parts[0])
    vec.append(parts[1])
    for i in range(1, 700):
        vec.append(str(i) + ":" + vec_m.get(i, "0"))
    print " ".join(vec)
