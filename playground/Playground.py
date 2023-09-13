from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase
from collections import defaultdict

d = defaultdict(lambda: 0)
facebookDb = FacebookDataDatabase()
ids = list(map(lambda x: x[0], facebookDb.get_post_ids()))
for id in ids:
    page_id = id.split("_")[0]
    d[page_id] += 1

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 4))
plt.subplots_adjust(wspace=0.3)

plt.figure(1)
plt.hist(d.values(), bins=30, color='g')
plt.ylabel('Bin Count');
plt.xlabel('Number of Posts Scraped');
plt.plot()

plt.subplots_adjust(hspace=.5)

plt.savefig('Posts_Per_Page_Histogram.png', bbox_inches='tight', dpi=300)
