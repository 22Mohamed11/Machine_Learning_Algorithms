import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tkinter.messagebox import *
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from  sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
import pandas as pd


# initalise the tkinter GUI
root = tk.Tk()
root.configure(bg='alice blue')

root.geometry("800x600") # set the root dimensions
root.pack_propagate(False) # tells the root to not let the widgets inside it determine its size.
root.resizable(0, 0)
# makes the root window fixed in size.

# Frame for TreeView
frame1 = tk.LabelFrame(root, text="Excel Data",bg='alice blue')
frame1.place(height=250, width=750)

# Frame for open file dialog
file_frame = tk.LabelFrame(root, text="Open File",bg='alice blue')
file_frame.place(height=100, width=400, rely=0.45, relx=0)

# Buttons
button1 = tk.Button(file_frame, text="Browse A File", command=lambda: File_dialog())
button1.place(rely=0.65, relx=0.50)

button2 = tk.Button(file_frame, text="Load File", command=lambda: Load_excel_data())
button2.place(rely=0.65, relx=0.30)

# The file/file path text
label_file = ttk.Label(file_frame, text="No File Selected")
label_file.place(rely=0, relx=0)


## Treeview Widget
tv1 = ttk.Treeview(frame1)
tv1.place(relheight=1, relwidth=1) # set the height and width of the widget to 100% of its container (frame1).

treescrolly = tk.Scrollbar(frame1, orient="vertical", command=tv1.yview) # command means update the yaxis view of the widget
treescrollx = tk.Scrollbar(frame1, orient="horizontal", command=tv1.xview) # command means update the xaxis view of the widget
tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set) # assign the scrollbars to the Treeview Widget
treescrollx.pack(side="bottom", fill="x") # make the scrollbar fill the x axis of the Treeview widget
treescrolly.pack(side="right", fill="y") # make the scrollbar fill the y axis of the Treeview widget


def File_dialog():
    """This Function will open the file explorer and assign the chosen file path to label_file"""
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select A File",
                                          filetype=(("csv files", "*.csv"),("All Files", "*.*")))
    label_file["text"] = filename
    return None


def Load_excel_data():
    """If the file selected is valid this will load the file into the Treeview"""
    file_path = label_file["text"]
    try:
        excel_filename = r"{}".format(file_path)
        if excel_filename[-4:] == ".csv":
            df = pd.read_csv(excel_filename)
        else:
            df = pd.read_excel(excel_filename)

    except ValueError:
        tk.messagebox.showerror("Information", "The file you have chosen is invalid")
        return None
    except FileNotFoundError:
        tk.messagebox.showerror("Information", f"No such file as {file_path}")
        return None

    clear_data()
    tv1["column"] = list(df.columns)
    tv1["show"] = "headings"
    for column in tv1["columns"]:
        tv1.heading(column, text=column) # let the column heading = column name

    df_rows = df.to_numpy().tolist() # turns the dataframe into a list of lists
    for row in df_rows:
        tv1.insert("", "end", values=row) # inserts each list into the treeview. For parameters see https://docs.python.org/3/library/tkinter.ttk.html#tkinter.ttk.Treeview.insert
    return None


def clear_data():
    tv1.delete(*tv1.get_children())
    return None

######################################################################################################################################################

#-------------------------------------------------- preprocessing ----------------------------------------------------------#
def openNewWindow():
    newWindow= tk.Toplevel(root)

    newWindow.title("preprocessing")
    newWindow.geometry("400x400")
    newWindow.configure(bg='alice blue')
    frame2 = tk.LabelFrame(newWindow, text=" Data")
    frame2.place(height=300, width=400)
    tv2 = ttk.Treeview(frame2)
    tv2.place(relheight=1, relwidth=1)  # set the height and width of the widget to 100% of its container (frame1).
    treescrolly = tk.Scrollbar(frame2, orient="vertical",
                               command=tv2.yview)  # command means update the yaxis view of the widget
    treescrollx = tk.Scrollbar(frame2, orient="horizontal",
                               command=tv2.xview)  # command means update the xaxis view of the widget
    tv2.configure(xscrollcommand=treescrollx.set,
                  yscrollcommand=treescrolly.set)  # assign the scrollbars to the Treeview Widget
    treescrollx.pack(side="bottom", fill="x")  # make the scrollbar fill the x axis of the Treeview widget
    treescrolly.pack(side="right", fill="y")  # make the scrollbar fill the y axis of the Treeview widget


    btn5=tk.Button(newWindow,text="IMPUTER",width=10,command=lambda :impute_data(),bg='green')
    btn5.place(relx=0.4,rely=0.8)
    btn6 = tk.Button(newWindow, text="L encoder",width=10 ,command=lambda: label_encoder(),bg='green')
    btn6.place(relx=0.1, rely=0.8)
    btn7 = tk.Button(newWindow, text="Scale",width=10,command=lambda: standard_scaler(),bg='green')
    btn7.place(relx=0.1, rely=0.9)
    btn8 = tk.Button(newWindow, text="PCA", width=10, command=lambda :PCA_2(),bg='green')
    btn8.place(relx=0.4, rely=0.9)
    btn9 = tk.Button(newWindow, text="RFE", width=10, command=lambda: Rfe(),bg='green')
    btn9.place(relx=0.7, rely=0.9)
    btn10 = tk.Button(newWindow, text="Clear", width=10, command=lambda: clear_data(),bg='red')
    btn10.place(relx=0.7, rely=0.8)

    def clear_data():
        tv2.delete(*tv2.get_children())

    def impute_data():
        imputer=SimpleImputer(strategy='most_frequent')
        mydata.iloc[:,0:8]=imputer.fit_transform(mydata.iloc[:,0:8])
        result_show()
    def label_encoder():
        le=LabelEncoder()
        mydata.iloc[:,8]=le.fit_transform(mydata.iloc[:,8])
        result_show()
    def standard_scaler():

        x = mydata.iloc[:, :-1].values
        y = mydata.iloc[:, -1].values
        st=StandardScaler(copy=True,with_mean=True,with_std=True)
        x=pd.DataFrame(st.fit_transform(x))
        tv2["column"] = list(x.columns)
        tv2["show"] = "headings"
        for column in tv2["columns"]:
            tv2.heading(column, text=column)  # let the column heading = column name

        x_rows = x.to_numpy().tolist()  # turns the dataframe into a list of lists
        for row in x_rows:
            tv2.insert("", "end",
                       values=row)  # inserts each list into the treevie
        return None

    def PCA_2():
        le = LabelEncoder()
        mydata.iloc[:, 8] = le.fit_transform(mydata.iloc[:, 8])

        X = mydata.drop(columns=['Age'], axis=1)
        y = mydata['Age']
        scaler = StandardScaler()
        Scaled_data_X = scaler.fit_transform(X)
        Scaled_data_X

        pca = PCA(n_components=2)
        pca_X = pca.fit_transform(Scaled_data_X)
        print("pca=:",pca_X)
        tv2["column"] = list(pca_X)
        tv2["show"] = "headings"
        for row in pca_X:
            tv2.insert("", "end",
                       values=row)  # inserts each list into the treeviewo
        return None
    def Rfe ():
        label_encoder()
        impute_data()

        X=mydata.drop(columns=['Age'],axis=1)
        y=mydata['Age']
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)
        sc=StandardScaler()
        X_train_std=sc.fit_transform(x_train)
        X_test_std=sc.transform(x_test)
        rfe=RFE(estimator=DecisionTreeClassifier(),n_features_to_select=3)
        rfe.fit(X_train_std,y_train)
        X_train_sub=rfe.transform(X_train_std)

        tv2["column"] = list(X_train_sub)
        tv2["show"] = "headings"
        for row in X_train_sub:
            tv2.insert("", "end",
                       values=row)  # inserts each list into the treeviewo
        return None

    def result_show():
        tv2["column"] = list(mydata.columns)
        tv2["show"] = "headings"
        for column in tv2["columns"]:
            tv2.heading(column, text=column)  # let the column heading = column name

        mydata_rows = mydata.to_numpy().tolist()  # turns the dataframe into a list of lists
        for row in mydata_rows:
            tv2.insert("", "end",
                       values=row)  # inserts each list into the treeview
        return None

###############################################################################################################################################

#--------------------------------------- REGRESSION -----------------------------------------------------------#
def window():
    newWindow = tk.Toplevel(root)
    newWindow.title("Regression")
    newWindow.geometry("400x400")
    newWindow.configure(bg='alice blue')
    frame3 = tk.LabelFrame(newWindow, text=" REG")
    frame3.place(height=150, width=140)
    tv3 = ttk.Treeview(frame3)
    tv3.place(relheight=1, relwidth=1)  # set the height and width of the widget to 100% of its container (frame1).
    treescrolly = tk.Scrollbar(frame3, orient="vertical",
                               command=tv3.yview)  # command means update the yaxis view of the widget

    tv3.configure(xscrollcommand=treescrollx.set,
                  yscrollcommand=treescrolly.set)  # assign the scrollbars to the Treeview Widget

    treescrolly.pack(side="right", fill="y")  # make the scrollbar fill the y axis of the Treeview widget
    btn10=tk.Button(newWindow,text='Liner Regression',width=15,command=lambda :reg())
    btn10.place(relx=0.25,rely=0.5)
    btn11 = tk.Button(newWindow, text='MSE', width=5, command=lambda: mse())
    btn11.place(relx=0.6, rely=0.5)
    mse_label = tk.Label(newWindow, text="Mean Square Error: -")
    mse_label.place(relx=0.65, rely=0.58)


    df=pd.read_csv('dataset.csv')
    X=df[['Time']]
    Y =df['radius']
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=10)
    regr=linear_model.LinearRegression()
    regr.fit(X_train,Y_train)
    lol=regr.predict(X_test)
    def reg():
         tv3["column"] = list(lol)
         tv3["show"] = "headings"
         for row in lol:
             tv3.insert("", "end",
                        values=row)  # inserts each list into the treeviewo

    def mse():
        mse = mean_squared_error(Y_test, lol)
        mse_label = tk.Label(newWindow, text="Mean Square Error: -")
        mse_label.place(relx=0.65, rely=0.65)
        mse_label.config(text=" {:.2f}".format(mse))

    return None

######################################################################################################################################################

#___________________________________________classification____________________________________________________________________________#
def wdo():
    lev=pd.read_csv('students.csv')
    le = LabelEncoder()
    lev.iloc[:, 6] = le.fit_transform(lev.iloc[:, 6])

    newWindow = tk.Toplevel(root)
    newWindow.title("Classification")
    newWindow.geometry("400x400")
    newWindow.configure(bg='alice blue')
    frame3 = tk.LabelFrame(newWindow, text=" Model")
    frame3.place(height=300, width=200,rely=0.2,relx=0.05)
    X=lev.iloc[:,:-1].values
    Y=lev.iloc[:,-1].values
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)
    def knn():
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        matrix = confusion_matrix(y_test, predictions)
        print(matrix)
        Acc = accuracy_score(y_test, predictions)
        acc_label = tk.Label(newWindow, text="")
        acc_label.place(relx=0.3, rely=0.40)
        acc_label.config(text=" {:.2f}".format(Acc))

        pre = precision_score(y_test, predictions)
        pre_label = tk.Label(newWindow, text="")
        pre_label.place(relx=0.3, rely=0.5)
        pre_label.config(text=" {:.2f}".format(pre))

        rec = recall_score(y_test, predictions)
        rec_label = tk.Label(newWindow, text="")
        rec_label.place(relx=0.3, rely=0.6)
        rec_label.config(text=" {:.2f}".format(rec))

        f1s = f1_score(y_test, predictions)
        f1s_label = tk.Label(newWindow, text="")
        f1s_label.place(relx=0.3, rely=0.7)
        f1s_label.config(text=" {:.2f}".format(f1s))
    def support ():
        classifiere = svm.SVC(kernel="rbf")
        classifiere.fit(x_train, y_train)
        predictions = classifiere.predict(x_test)
        matrix = confusion_matrix(y_test, predictions)
        print(matrix)
        Acc = accuracy_score(y_test, predictions)
        acc_label = tk.Label(newWindow, text="Mean Square Error: -")
        acc_label.place(relx=0.3, rely=0.40)
        acc_label.config(text=" {:.2f}".format(Acc))

        pre = precision_score(y_test, predictions)
        pre_label = tk.Label(newWindow, text="Mean sqyare Error:-")
        pre_label.place(relx=0.3, rely=0.5)
        pre_label.config(text=" {:.2f}".format(pre))
        print(pre)

        rec = recall_score(y_test, predictions)
        rec_label = tk.Label(newWindow, text="Mean square Error:-")
        rec_label.place(relx=0.3, rely=0.6)
        rec_label.config(text=" {:.2f}".format(rec))

        f1s = f1_score(y_test, predictions)
        f1s_label = tk.Label(newWindow, text="Mean square Error:-")
        f1s_label.place(relx=0.3, rely=0.7)
        f1s_label.config(text=" {:.2f}".format(f1s))

    btn22=tk.Button(newWindow,text="KNN",width=10,command=knn)
    btn22.place(relx=0.6,rely=0.7)
    btn33 = tk.Button(newWindow, text="SVM", width=10, command=support)
    btn33.place(relx=0.6, rely=0.8)
    l4=tk.Label(newWindow,text="Accuracy:",width=10)
    l4.place(relx=0.1,rely=0.4)
    l1 = tk.Label(newWindow, text="Precision:", width=10)
    l1.place(relx=0.1, rely=0.5)
    l2 = tk.Label(newWindow, text="Recall:", width=10)
    l2.place(relx=0.1, rely=0.6)
    l3 = tk.Label(newWindow, text="F1_Score:", width=10)
    l3.place(relx=0.1, rely=0.7)

btn=tk.Button(root,text="Preprocessing",command=openNewWindow,width=15,height=3,bg="black",fg="white",font=26)
btn.place(rely=0.68,relx=0.70)
btn2=tk.Button(root,text="Regression",width=15,height=3,bg="black",fg="white",command=window,font=26)
btn2.place(rely=0.68,relx=0.40)
btn3=tk.Button(root,text="Classlfication",width=15,height=3,bg="black",fg="white",command=wdo,font=26)
btn3.place(rely=0.68,relx=0.10)
mydata=pd.read_csv("abalone_train (1).csv")

root.mainloop()