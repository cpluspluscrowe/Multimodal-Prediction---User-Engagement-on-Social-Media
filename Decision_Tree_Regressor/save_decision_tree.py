import pydotplus
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import os
import sys

sys.path.append(os.pardir)
from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase

facebookDb = FacebookDataDatabase()
import pickle
from sklearn import tree


class Static:
    metric_getter = facebookDb.getShareCount


def get_model():
    model_name = Static.metric_getter.__name__ + "_regression.pkl"
    if os.path.exists(model_name):
        with open(model_name, 'rb') as pickle_file:
            return pickle.load(pickle_file)


if __name__ == '__main__':
    dot_data = StringIO()
    model = get_model()
    if model:
        from sklearn.externals.six import StringIO
        import pydot

        dot_data = StringIO()
        tree.export_graphviz(model, out_file=dot_data)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        graph[0].write_pdf("./{0}_combined_decision_tree.pdf".format(Static.metric_getter.__name__))
        #        tree.export_graphviz(model, out_file='tree.dot')
        # export_graphviz(model, out_file=dot_data,
        #                filled=True, rounded=True,
        #       #                special_characters=True)
        #      graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        #     Image(graph.create_png())
        print("Done")
