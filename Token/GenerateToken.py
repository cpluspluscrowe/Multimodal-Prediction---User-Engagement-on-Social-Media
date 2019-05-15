import urllib
import ast
from urllib import request


def getTokenData():
    account_data = [['1995698614010450', '967426a716717cd40e89f230f42a4466']]  # ['138310270170693',
    #                     '6ad13fe454637881b1d6fab97f1ba435']
    while True:
        for token_data in account_data:
            yield token_data


token_generator = getTokenData()


def getToken():
    with open("../Token/token.txt", "r") as f:
        text = f.read()
        return text
        #    token_data = next(token_generator)
        #    resp = urllib.request.urlopen(
        #        'https://graph.facebook.com/oauth/access_token?client_id={0}&client_secret={1}&grant_type=client_credentials'.format(token_data[0],token_data[1]))
        #    token = ast.literal_eval(resp.read().decode("utf-8"))["access_token"]
        # return token


#    return "EAACEdEose0cBAJFm6Mj52lZB7ZCykpFXq8ZAk1TKFiBmVcPOAjA54UOiiiccVBPmgZBnY35CEYv19TgsA3NDnraBCBP2NXGsX6SLXe7bXD9g1jJNNg0428DxgJJWCrzKvzTwczHEQLaGMyZCUsJBCSQznhgSiO9LwNNrfhEgGp0dYmH0ZAFfQePhmViTQFUA7JuBZBLblMNZCwZDZD"

if __name__ == "__main__":
    for x in range(10):
        print(next(token_generator))
    print()
    print(getToken())
