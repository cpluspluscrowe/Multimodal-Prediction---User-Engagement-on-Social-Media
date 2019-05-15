
def escape_url(url):
    if type(url) == str:
        escaped_image_url = url.replace("\/","/")
        return escaped_image_url
    else:
        return url

def get_key_data(data,key_name, ref_to_storage_function):
    if type(data) == list:
        if data:
            data = data[0]
            get_key_data(data, key_name, ref_to_storage_function)
    if type(data) == dict:
        for key in data:
            if key == key_name:
                escaped_url = escape_url(data[key])
                ref_to_storage_function(escaped_url)
            else:
                get_key_data(data[key], key_name, ref_to_storage_function)