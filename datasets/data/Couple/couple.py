#!/usr/bin/python3
#
#   generate special data that has strong feature combinations.
#

import random
import sys
import json

size_data = int(sys.argv[1]) if int(sys.argv[1]) != -1 else 10000
num_fc = 10
lens_fc = [ 3 ]
fields = [2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5]
max_repeat = 1

all_fc = []
all_data = []
all_targets = []
all_tags = []
fid = []


#
#   generate feature combinations
#
def gen_fc() :
    id_clock = 0
    for l in fields :
        cur = [0]
        for j in range(l) :
            id_clock = id_clock + 1
            cur.append(id_clock)
        fid.append(cur)
    for i in range(num_fc) :
        length = random.choice(lens_fc)
        sub_fields = random.sample(range(len(fields)), length)
        sorted(sub_fields)
        fc = dict()
        for f in sub_fields :
            fc[f] = random.randint(1, fields[f])
        all_fc.append(fc)
    return all_fc

def gen_data() :
    while len(all_data) < size_data :
        fc_id = random.randint(0, len(all_fc) - 1)
        fc = all_fc[fc_id]

        repeat = random.randint(1, min(size_data - len(all_data), len(fc)))
        repeat = min(repeat, max_repeat)
        for _ in range(repeat) :
            data = dict()
            for i in range(len(fields)) :
                if i in fc :
                    data[i] = fc[i]
                else :
                    data[i] = random.randint(1, fields[i])
            all_data.append(data)
            all_tags.append((fc_id, -1))

            for (p,v) in fc.items() :
                data = dict()
                for i in range(len(fields)) :
                    if i in fc and i != p :
                        data[i] = fc[i]
                    else :
                        data[i] = random.randint(1, fields[i])
                        if i in fc and i == p :
                            while data[i] == fc[i] :
                                data[i] = random.randint(1, fields[i])
                all_data.append(data)
                all_tags.append((fc_id, fid[p][v]))


def eval_targets() :
    for data in all_data :
        ok = False
        for fc in all_fc :
            subok = True
            for (p,v) in fc.items() :
                if data[p] != v :
                    subok = False
                    break
            if subok :
                ok = True
                break
        all_targets.append(ok)

def output() :
    with open("meta.txt", "w") as logfile:
        meta = dict()
        meta["field_sizes"] = fields
        meta["field_combinations"] = []
        for fc in all_fc:
            meta["field_combinations"].append([])
            for (p,v) in sorted(fc.items()) :
                meta["field_combinations"][-1].append(p)
        meta["all_fc"] = all_fc
        meta["lens_fc"] = lens_fc
        json.dump(meta, logfile, indent=2, sort_keys=True)
#                log.write("%d " % p)
#            log.write("\n")
    for i in range(len(all_data)) :
        print(1 if all_targets[i] else 0, end=" ")
        data = all_data[i]
        s = 0
        for j in range(len(fields)) :
            print("%s" % (str(s + data[j] - 1) + ":1 "), end="")
            s = s + fields[j]
#        print("%d %d" % (all_tags[i][0], all_tags[i][1]), end="")
        print()

def main() :
    gen_fc()
    gen_data()
    eval_targets()
    output()

if __name__ == "__main__" : 
    main()
