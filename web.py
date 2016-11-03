import os
import MySQLdb
import datetime
from flask import Flask, render_template, url_for, request, redirect, make_response, session

from FeatureSelection import feature_selection
from LayerTwo import LayerTwo
from LayerThree import LayerThree
from LayerFour import LayerFour

app = Flask(__name__)
app.secret_key = 'ccloveskkkklovescc'
imagepath = os.path.join(os.getcwd(), 'static/images')
filepath = os.path.join(os.getcwd(), 'static/uploadfiles')

@app.route('/', methods = ['GET', 'POST'])
def login():
	if request.method == 'POST':
		username = request.form.get('username')
		password = request.form.get('password')
		conn = MySQLdb.connect(user = 'root', passwd = '', host = '127.0.0.1')
		conn.select_db('fsweb')
		curr = conn.cursor()
		sql = "select password from user where username = '%s'" % username
		try:
			curr.execute(sql)
			result = curr.fetchone()
			password_db = result[0]
		except:
			print "Error: unable to fecth data"
			password_db = ''
		curr.close()
		conn.close()
		if password == password_db and password_db != '':
			response = make_response(redirect('/upload/'))
			response.set_cookie('username', value = username, max_age = 300)
			session['islogin'] = '1'
			return response
		else:
			session['islogin'] = '0'
			warnings = 'Wrong username or password!'
			return render_template('login.html', warnings = warnings)
	else:
		return render_template('login.html')
	
@app.route('/upload/', methods = ['GET', 'POST'])
def upload():
	sidenavlist = ['upload', 'delete']
	conn = MySQLdb.connect(user = 'root', passwd = '', host = '127.0.0.1')
	conn.select_db('fsweb')
	if request.method == 'POST':
		username = request.cookies.get('username')
		file = request.files['file']
		filename = file.filename
		if filename != '':
			dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
			curr = conn.cursor()
			sql = "insert into uploadfile(username, filename, date) values ('%s', '%s', '%s')" % (username, filename, dt)
			curr.execute(sql)
			conn.commit()
			curr.close()
			file.save(os.path.join(filepath, filename))
	curr = conn.cursor()
	sql = "select * from uploadfile"
	curr.execute(sql)
	results = curr.fetchall()
	curr.close()
	conn.close()
	return render_template('upload.html', sidenavlist = sidenavlist, results = results)

@app.route('/delete/', methods = ['GET', 'POST'])
def delete():
	sidenavlist = ['upload', 'delete']
	conn = MySQLdb.connect(user = 'root', passwd = '', host = '127.0.0.1')
	conn.select_db('fsweb')
	if request.method == 'POST':
		filename = request.form.get('filename')
		if filename != '':
			curr = conn.cursor()
			sql = "select * from uploadfile where filename = '%s'" % filename
			curr.execute(sql)
			results = curr.fetchall()
			curr.close()
			if results != ():
				curr = conn.cursor()
				sql = "delete from uploadfile where filename = '%s'" % filename
				curr.execute(sql)
				conn.commit()
				curr.close()
				print 'delete process'
				os.remove(os.path.join(filepath, filename))
	curr = conn.cursor()
	sql = "select * from uploadfile"
	curr.execute(sql)
	results = curr.fetchall()
	curr.close()
	conn.close()
	return render_template('delete.html', sidenavlist = sidenavlist, results = results)
	
@app.route('/prepare/', methods = ['GET', 'POST'])
def prepare():
	sidenavlist = ['prepare', 'layer1', 'layer2', 'layer3', 'layer4', 'whole']
	conn = MySQLdb.connect(user = 'root', passwd = '', host = '127.0.0.1')
	conn.select_db('fsweb')
	curr = conn.cursor()
	sql = "select * from uploadfile"
	curr.execute(sql)
	results = curr.fetchall()
	curr.close()
	if request.method == 'POST':
		filename = request.form.get('filename')
		curr = conn.cursor()
		sql = "select * from uploadfile where filename = '%s'" % filename
		curr.execute(sql)
		results = curr.fetchall()
		curr.close()
		if results == ():
			conn.close()
			return redirect('/prepare/')
		else:
			conn.close()
			session['filename'] = filename
			feature_selection(os.getcwd() + '/static/uploadfiles/' + filename)
			return redirect('/layer1/')
	conn.close()
	return render_template('prepare.html', sidenavlist = sidenavlist, results = results)

@app.route('/layer1/', methods = ['GET', 'POST'])
def layer1():
	sidenavlist = ['prepare', 'layer1', 'layer2', 'layer3', 'layer4', 'whole']
	with open(os.getcwd() + '/static/results/result1.txt', 'r') as f:
		results = f.read()
	if request.method == 'POST':
		LayerTwo(session.get('filename'))
		return redirect('/layer2/')
	return render_template('layer1.html', sidenavlist = sidenavlist, results = results)
	
@app.route('/layer2/', methods = ['GET', 'POST'])
def layer2():
	sidenavlist = ['prepare', 'layer1', 'layer2', 'layer3', 'layer4', 'whole']
	with open(os.getcwd() + '/static/results/result2.txt', 'r') as f:
		results = f.read()
	if request.method == 'POST':
		LayerThree(session.get('filename'))
		return redirect('/layer3/')
	return render_template('layer2.html', sidenavlist = sidenavlist, results = results)

@app.route('/layer3/', methods = ['GET', 'POST'])
def layer3():
	sidenavlist = ['prepare', 'layer1', 'layer2', 'layer3', 'layer4', 'whole']
	with open(os.getcwd() + '/static/results/result3.txt', 'r') as f:
		results = f.read()
	if request.method == 'POST':
		LayerFour()
		return redirect('/layer4/')
	return render_template('layer3.html', sidenavlist = sidenavlist, results = results)

@app.route('/layer4/', methods = ['GET', 'POST'])
def layer4():
	sidenavlist = ['prepare', 'layer1', 'layer2', 'layer3', 'layer4', 'whole']
	with open(os.getcwd() + '/static/results/result4.txt', 'r') as f:
		results = f.read()
	if request.method == 'POST':
		return redirect('/whole/')
	return render_template('layer4.html', sidenavlist = sidenavlist, results = results)

@app.route('/whole/')
def whole():
	sidenavlist = ['prepare', 'layer1', 'layer2', 'layer3', 'layer4', 'whole']
	with open(os.getcwd() + '/static/results/result.txt', 'r') as f:
		results = f.read()
	return render_template('whole.html', sidenavlist = sidenavlist, results = results)
	
@app.route('/user/')
def user():
	sidenavlist = ['change_username', 'change_password']
	account = request.cookies.get('username')
	return render_template('user.html', sidenavlist = sidenavlist, account = account)
	
@app.route('/change_username/', methods = ['GET', 'POST'])
def change_username():
	sidenavlist = ['change_username', 'change_password']
	account = request.cookies.get('username')
	if request.method == 'POST':
		username = request.form.get('username')
		conn = MySQLdb.connect(user = 'root', passwd = '', host = '127.0.0.1')
		conn.select_db('fsweb')
		curr = conn.cursor()
		sql = "update user set username = '%s' where username = '%s'" % (username, account)
		curr.execute(sql)
		conn.commit()
		curr.close()
		conn.close()
		response = make_response(redirect('/'))
		response.set_cookie('username', value = '', max_age = 300)
		session['islogin'] = '0'
		return response
	return render_template('change_username.html', sidenavlist = sidenavlist)

@app.route('/change_password/', methods = ['GET', 'POST'])
def change_password():
	sidenavlist = ['change_username', 'change_password']
	account = request.cookies.get('username')
	if request.method == 'POST':
		password = request.form.get('password')
		conn = MySQLdb.connect(user = 'root', passwd = '', host = '127.0.0.1')
		conn.select_db('fsweb')
		curr = conn.cursor()
		sql = "update user set password = '%s' where username = '%s'" % (password, account)
		curr.execute(sql)
		conn.commit()
		curr.close()
		conn.close()
		response = make_response(redirect('/'))
		response.set_cookie('username', value = '', max_age = 300)
		session['islogin'] = '0'
		return response
	return render_template('change_password.html', sidenavlist = sidenavlist)

if __name__ == '__main__':
	app.run()