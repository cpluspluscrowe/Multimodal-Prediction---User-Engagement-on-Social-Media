# coding: utf-8

# In[ ]:


import os
import sys

import requests

sys.path.append(os.pardir)
from bs4 import BeautifulSoup as BS
from Notebooks.LinkDatabases.Expresso import *  # expressoDb
from Notebooks.LinkDatabases.FbAd import *  # adDb


def getFacebookLinks(urls):
    for url in urls:
        url = url[0]
        url = url.split("?")[0]
        if not adDb.hasEspressoUrl(url):
            response = requests.get(url)
            soup = BS(response.text, "lxml")
            widget = soup.findAll("div", {"class": "fb-post"})
            if widget:
                link = widget[0]["data-href"]
                print("Url:", link)
                adDb.insertAdLinks([url], link)
            else:
                print("No")


print("Number of fb urls in db:", len(expressoDb.selectExpressoData()))
facebook_urls = expressoDb.selectExpressoData()
getFacebookLinks(facebook_urls)
