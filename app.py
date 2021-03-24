from flask import Flask, request,render_template
import BBC

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/submit', methods = ['POST'])
def getText():
    text,label = BBC.process_url(request.form['url_name'])
    print(text,label)
    return render_template('home.html',text=text, label=label)


if __name__ == '_main_':
    app.run(debug=True)
