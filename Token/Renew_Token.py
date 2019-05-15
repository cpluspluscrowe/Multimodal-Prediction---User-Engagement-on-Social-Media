# coding: utf-8

# In[1]:


import pickle
import sys
import os

sys.path.append(os.pardir)
# from Notebooks.LinkDatabases.Expresso import expressoDb
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

from selenium import webdriver
import time


def renew_token():
    url = "https://www.facebook.com/login/?next=https%3A%2F%2Fdevelopers.facebook.com%2Ftools%2Fexplorer%2F145634995501895%2F"
    # setup chrome driver
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(
        "/usr/local/bin/chromedriver",
        chrome_options=chrome_options)
    # driver = webdriver.PhantomJS(executable_path=r"/home/ailab/ccrowe/github/phantomjs-2.1.1-linux-x86_64/bin/phantomjs")
    hrefs = []
    driver.get(url)
    elem = driver.find_element_by_id("email")
    elem.send_keys("cpluspluscrowe")
    elem = driver.find_element_by_id("pass")
    elem.send_keys("FakePassword")
    elem = driver.find_element_by_id("loginbutton")
    elem.send_keys(Keys.RETURN)

    driver.get("https://developers.facebook.com/tools/explorer/145634995501895/")

    vals = driver.find_elements_by_class_name("_58al")
    code = ""
    for val in vals:
        text = val.get_attribute('value')
        if len(text) > 150:
            code = text

    print(code)
    driver.quit()

    with open("token.txt", "w") as f:
        f.write(code)
        print("Done writing code")


import time

while True:
    starttime = time.time()
    renew_token()
    time.sleep(300)

if __name__ == "__main__":
    pass
