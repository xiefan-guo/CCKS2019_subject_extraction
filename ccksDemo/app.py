from flask import Flask, render_template, request, flash,  jsonify, redirect, url_for, session
from utils import extract_entity

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def test():
    if request.method == 'GET':
        result = 'NaN'
        return render_template('test.html', result=result)
    else:
        comments = request.form.get('comments')
        print(comments)
        lst = comments.split(',', 1)

        print(lst)

        s = u'"%s"' % (extract_entity.extract_entity_self(lst[0].replace('\t', ''), lst[1]))
        print(s)
        return render_template('test.html', result=s)


if __name__ == '__main__':
    app.run()
