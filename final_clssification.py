import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import NaiveBayesClassifier
import os

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

# Function to extract keywords from a file
def extract_keywords(file_path):
    with open(file_path, "r") as f:
        file_contents = f.read()
        tokens = word_tokenize(file_contents.lower())
        keywords = [word for word in tokens if word.isalnum() and word not in stop_words]
        output_string = ' '.join(keywords)
    return output_string

# Function to classify the department based on keywords
def classify_department(file_content):
    department = classifier.classify(dict([(feature, (feature in word_tokenize(file_content.lower()))) for feature in features]))
    return department

# Function to select a file
def select_file():
    global file_path
    file_path = filedialog.askopenfilename()
    entry_file_path.delete(0, tk.END)
    entry_file_path.insert(0, file_path)

# Function to select the test folder
def select_test_folder():
    global test_folder_path
    test_folder_path = filedialog.askdirectory()
    entry_test_folder_path.delete(0, tk.END)
    entry_test_folder_path.insert(0, test_folder_path)

# Function to extract keywords from the selected file and display them in the Treeview
def keywordise():
    file_content = extract_keywords(file_path)
    department = classify_department(file_content)
    tree.insert("", tk.END, values=(file_path, file_content, department))

# Function to calculate the accuracy of the classifier on test data
def calculate_accuracy():
    global classifier
    folder_path = test_folder_path
    test_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            department = None

            if filename.startswith("hr_"):
                department = "HR"
            elif filename.startswith("fin_"):
                department = "Finance"
            elif filename.startswith("rand_"):
                department = "R&D"

            if department is not None:
                file_content = extract_keywords(file_path)
                features_dict = dict([(feature, (feature in word_tokenize(file_content.lower()))) for feature in features])
                test_data.append((features_dict, department))

    accuracy = nltk.classify.accuracy(classifier, test_data)
    accuracy_percentage = round(accuracy * 100, 2)
    accuracy_label.config(text="Accuracy: {}%".format(accuracy_percentage))

    # Update training data with correct predictions
    for features_dict, department in test_data:
        predicted_department = classifier.classify(features_dict)
        if predicted_department == department and (features_dict, department) not in training_data:
            training_data.append((features_dict, department))
    # Updated Classifier        
    classifier = NaiveBayesClassifier.train(training_data)  
    # Print the number of key value pairs in training data after the update
    keyword_count = sum(len(features_dict) for features_dict, _ in training_data)
    print("Total number of key-value pairs in training data after update: ", keyword_count)

# Function to extract features from a document
def document_features(document):
    words = set(document)
    features = {}
    word_features = []
    for word in word_features:
        features['contains({})'.format(word)] = (word in words)
    return features

# Create a list to store keywords for each department
department_keywords = {
    "Finance": [],
    "HR": [],
    "R&D": []
}

# Process files in the finance folder
path = 'C:\\Users\\hp\\OneDrive\\Desktop\\fin_train'
for filename in os.listdir(path):
    if filename.endswith(".txt"):
        file_path = os.path.join(path, filename)
        with open(file_path, 'r') as file:
            file_contents = file.read()
            finance_words = word_tokenize(file_contents)
            finance_keywords = [w.lower() for w in finance_words if w not in stop_words]
            department_keywords["Finance"].extend(finance_keywords)

# Process files in the HR folder
path = 'C:\\Users\\hp\\OneDrive\\Desktop\\hr_train'
for filename in os.listdir(path):
    if filename.endswith(".txt"):
        file_path = os.path.join(path, filename)
        with open(file_path, 'r') as file:
            file_contents = file.read()
            hr_words = word_tokenize(file_contents)
            hr_keywords = [w.lower() for w in hr_words if w not in stop_words]
            department_keywords["HR"].extend(hr_keywords)

# Process files in the R&D folder
path = 'C:\\Users\\hp\\OneDrive\\Desktop\\r&d_train'
for filename in os.listdir(path):
    if filename.endswith(".txt"):
        file_path = os.path.join(path, filename)
        with open(file_path, 'r') as file:
            file_contents = file.read()
            randd_words = word_tokenize(file_contents)
            randd_keywords = [w.lower() for w in randd_words if w not in stop_words]
            department_keywords["R&D"].extend(randd_keywords)

# Combine all department keywords
all_keywords = []
for keywords in department_keywords.values():
    all_keywords.extend(keywords)

features = [keyword for keyword in all_keywords]

# Create the training data using the labeled files
training_data = []
department_folders = {
    "Finance": "C:\\Users\\hp\\OneDrive\\Desktop\\fin_train",
    "HR": "C:\\Users\\hp\\OneDrive\\Desktop\\hr_train",
    "R&D": "C:\\Users\\hp\\OneDrive\\Desktop\\r&d_train"
}
for department, folder_path in department_folders.items():
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            file_content = extract_keywords(file_path)
            features_dict = dict([(feature, (feature in word_tokenize(file_content.lower()))) for feature in features])
            training_data.append((features_dict, department))

# Train the Naive Bayes classifier
classifier = NaiveBayesClassifier.train(training_data)

# GUI setup
window = tk.Tk()
window.title("Department Classifier")
window.geometry("800x600")

# File selection frame
frame_select = ttk.Frame(window)
frame_select.pack(pady=10)

label_file_path = ttk.Label(frame_select, text="File Path:")
label_file_path.grid(row=0, column=0, padx=10, sticky="W")

entry_file_path = ttk.Entry(frame_select, width=60)
entry_file_path.grid(row=0, column=1, padx=10, sticky="W")

button_select_file = ttk.Button(frame_select, text="Select File", command=select_file)
button_select_file.grid(row=0, column=2, padx=10)

# Test folder selection frame
frame_test_folder = ttk.Frame(window)
frame_test_folder.pack(pady=10)

label_test_folder_path = ttk.Label(frame_test_folder, text="Test Folder Path:")
label_test_folder_path.grid(row=0, column=0, padx=10, sticky="W")

entry_test_folder_path = ttk.Entry(frame_test_folder, width=60)
entry_test_folder_path.grid(row=0, column=1, padx=10, sticky="W")

button_select_test_folder = ttk.Button(frame_test_folder, text="Select Test Folder", command=select_test_folder)
button_select_test_folder.grid(row=0, column=2, padx=10)

# Keyword extraction and classification frame
frame_classification = ttk.Frame(window)
frame_classification.pack(pady=10)

button_keywordise = ttk.Button(frame_classification, text="Keywordise", command=keywordise)
button_keywordise.grid(row=0, column=0, padx=10)

# Treeview to display the extracted keywords and department
tree = ttk.Treeview(frame_classification)
tree["columns"] = ("File Path", "Keywords", "Department")
tree.column("#0", width=0, stretch=tk.NO)
tree.column("File Path", anchor=tk.W, width=200)
tree.column("Keywords", anchor=tk.W, width=400)
tree.column("Department", anchor=tk.W, width=100)
tree.heading("File Path", text="File Path", anchor=tk.W)
tree.heading("Keywords", text="Keywords", anchor=tk.W)
tree.heading("Department", text="Department", anchor=tk.W)
tree.grid(row=1, column=0, padx=10)

# Accuracy calculation frame
frame_accuracy = ttk.Frame(window)
frame_accuracy.pack(pady=10)

button_calculate_accuracy = ttk.Button(frame_accuracy, text="Calculate Accuracy", command=calculate_accuracy)
button_calculate_accuracy.grid(row=0, column=0, padx=10)

accuracy_label = ttk.Label(frame_accuracy, text="Accuracy: ")
accuracy_label.grid(row=0, column=1, padx=10)

window.mainloop()
