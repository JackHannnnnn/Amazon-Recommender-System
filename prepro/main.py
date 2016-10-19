import gzip
import os
from IPython import embed

home = os.path.expanduser("~")
data_path = os.path.join(home, "data", "amazon.txt.gz")

def parse(field_list = ['productId', 'score', 'time', 'userId'], limit=2000000):
    """
    args field_list : list of fields to return
	(price, productId, title, helpfulness, profileName, score, summary, text, time, userId)
    args limit : limit of entity to store, None if you don't want limit
	(but since data is so big, so might get core dump)

    return: list of entry (dictionary with keys of field_list)
    """

    f = gzip.open(data_path, 'r')
    entry_list = []
    entry = {}
    for l in f:
    	l = l.strip()
    	colonPos = l.find(':')
    	if colonPos == -1:
	    entry_list.append(entry)
	    if len(entry_list)==limit : break
      	    entry = {}
      	    continue
        eName = l[:colonPos].split('/')[1]
	if eName not in field_list:
	    continue
        rest = l[colonPos+2:]
        entry[eName] = rest
    return entry_list


entry_list = parse()
print len(entry_list)
embed()



