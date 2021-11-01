from numpy import vectorize
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':

    corpus=["he is a good boy and he is most inetlligent boy","i'm going to doodly and see be to going and see"]
    vectorize=CountVectorizer()
    x=vectorize.fit_transform(corpus)
    print(vectorize.get_feature_names_out())
    print(x.toarray())

