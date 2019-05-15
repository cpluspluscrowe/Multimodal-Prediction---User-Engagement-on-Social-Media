#Setup Facebook API graph object
import facebook
import requests
import ast
from pprint import pprint
import sys
sys.path.append("..")
from Token.GenerateToken import getToken
token = getToken()

class PageMetric():
    def __init__(self):
        #-1 denotes that no value was found for this field
        self.about = ""
        self.fan_count = -1
        self.talking_about_count = -1
        self.rating_count = -1
        self.star_rating = -1

def escape_url(url):
    if type(url) == str:
        escaped_image_url = url.replace("\/","/")
        return escaped_image_url
    else:
        return url

def getPageMetricDataUrl(url):
    #kept vars in method to prevent scope pollution
    graph_url_start = r"https://graph.facebook.com/v2.11/"
    end_token = r'''?access_token={0}&fields=about,fan_count,talking_about_count,rating_count,overall_star_rating&debug=all&format=json&method=get&pretty=0&suppress_http_code=1000'''.format(
        token)
    id1 = url[url.find(graph_url_start) + len(graph_url_start):url.find(r"?access_token")]
    page_url = graph_url_start + id1 + end_token
    return page_url

def getPageMetricDict(url_to_request_for_data):
    data = requests.get(url_to_request_for_data)
    page_metric = PageMetric()
    try:
        metric_data = ast.literal_eval(data.text)
    except Exception as e:
        print(e,url_to_request_for_data)
        return page_metric
    page_metric.about = metric_data.get("about", None)
    page_metric.fan_count = metric_data.get("fan_count", None)
    page_metric.talking_about_count = metric_data.get("talking_about_count", None)
    page_metric.rating_count = metric_data.get("rating_count", None)
    page_metric.star_rating = metric_data.get("overall_star_rating", None)
    return page_metric

def getPageMetrics(page_url):
    page_url = escape_url(page_url)
    data_url = getPageMetricDataUrl(page_url)
    metrics = getPageMetricDict(data_url)
    return metrics

# How to use:
# Call select data and use those returned urls

if __name__ == "__main__":
    url = ""
    page_url = "https://graph.facebook.com/v2.11/2021081018168674?access_token=1995698614010450|FPRc6najDBjIsr-RQQFWheFwOHQ&fields=posts.limit(1000)&debug=all&format=json&method=get&pretty=0&suppress_http_code=1000"
    metrics = getPageMetrics(page_url)
    pprint(vars(metrics))









