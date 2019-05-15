# coding: utf-8

# In[1]:


import os
import pickle
import sys

sys.path.append(os.pardir)
from Notebooks.LinkDatabases.Expresso import expressoDb
from selenium.webdriver.chrome.options import Options


def storeSet(l):
    file = open(r'pagesVisited.pkl', 'wb')
    pickle.dump(l, file)
    file.close()


def reloadSet():
    file = open(r'pagesVisited.pkl', 'rb')
    l = pickle.load(file)
    file.close()
    return l


# storeSet(set()) #resets the value
visited = reloadSet()
print(visited)

from selenium import webdriver
import time

url = "https://adespresso.com/ads-examples/?login=1&r=5989608317"


def setupChromeDriver():
    # setup chrome driver
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome("../../chromedriver", chrome_options=chrome_options)
    hrefs = []
    driver.get(url)

    # login
    login = driver.find_element_by_class_name("newsletter-inline-widget")
    inputs = login.find_elements_by_tag_name("input")
    for x in inputs:
        try:
            x.click()
            x.send_keys("jacobjill@gmail.com")
            x.click()
            time.sleep(7)
        except:
            pass
    return driver


def getPageNumber(driver, current_page=-1):
    pageNumber = int(driver.execute_script(
        '''var pageNumber = document.getElementsByClassName("pagination-info").item(0).innerHTML;return pageNumber;''').replace(
        "Page <strong>", "").split("</strong>")[0].replace(" ", ""))
    if current_page == pageNumber:
        time.sleep(0.0001)
        return getPageNumber(driver, current_page)
    return pageNumber


def addToSet(pageNumber):
    visited = reloadSet()
    visited.add(pageNumber)
    storeSet(visited)
    return visited


def storePage(driver):
    pageNumber = getPageNumber(driver)
    return addToSet(pageNumber)


def gotoNextPage(driver):
    visited = storePage(driver)
    previous_page = getPageNumber(driver)

    driver.execute_script('''var x = document.getElementsByClassName("pagination-prev");x[0].click();''')
    pageNumber = getPageNumber(driver, previous_page)
    if pageNumber in visited:
        print("Skip:", pageNumber)
        gotoNextPage(driver)
    else:
        print("New", pageNumber)


def getAdPages(driver):
    try:
        hrefs = []
        time.sleep(2)
        items = driver.find_elements_by_class_name("sf-item")
        for item in items:
            link = item.find_element_by_tag_name('a')
            href = link.get_attribute('href')
            hrefs.append(href)
        expressoDb.insertExpressoLinks(hrefs)
        gotoNextPage(driver)
        getAdPages(driver)
    except:
        getAdPages(driver)


def run():
    reloadSet()
    driver = setupChromeDriver()
    getAdPages(driver)


run()

if __name__ == "__main__":
    pass
