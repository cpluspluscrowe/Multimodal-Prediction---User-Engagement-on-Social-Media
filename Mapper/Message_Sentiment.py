from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase


class MessageGetter:
    facebookDb = FacebookDataDatabase()

    @staticmethod
    def get_columns(remove_columns=True):
        columns = MessageGetter.facebookDb.getColumnNames()
        columns = list(map(lambda x: x[1], columns))
        if remove_columns:
            columns.remove("imageId")
            columns.remove("imageUrl")
        return columns

    @staticmethod
    def __dict_factory(row, columns):
        d = {}
        for idx, col in enumerate(columns):
            d[col] = row[idx]
        return d

    @staticmethod
    def __get_post():
        number_of_posts_to_train_on = len(MessageGetter.facebookDb.getFacebookDataWithPositiveCommentCounts())
        postData = MessageGetter.facebookDb.getFacebookDataWithPositiveCommentCounts()
        for i, post in enumerate(postData[:number_of_posts_to_train_on]):
            if i % 1000 == 0:
                print("gather data percent: ", i / number_of_posts_to_train_on)
            post_obj = MessageGetter.__dict_factory(post, MessageGetter.get_columns(False))
            yield post_obj

    @staticmethod
    def get_post_generator():
        post_generator = MessageGetter.__get_post()
        return post_generator
